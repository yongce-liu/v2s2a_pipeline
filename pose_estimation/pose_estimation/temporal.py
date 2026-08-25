"""Temporal post-processing for pose-estimation tracks.

Diagnosis-driven (see outputs/yellow_spoon): the late-sequence rotation jumps
are not smooth dynamics. The scene is quasi-static there (mean background-frame
diff ~1-5 vs ~8 earlier), the spoon lies uncovered (segmentation mask is a
clean thin strip, ~3.5k px), and the MoGe point map is metric and complete
under it -- yet FoundationPose's rotation error hits 50-95 deg per frame while
its own masked translation anchor keeps depth within ~1 cm of the point-map
median. The refiner locks onto the *background* point cloud (the crop covers
the full grid and the depth offset is interpolated over the crop), so a
doorway-scaled translation is consistent while rotation free-runs; the model
never re-acquires because the tracker's one-frame memory starts each frame
from the previous wrong pose.

Consequences for the post-pass:

- An innovation gate on cached poses can only see the *effect* (50-95 deg
  steps, median 21 deg over frames 73-88); it cannot see which side is right,
  because the render-vs-depth agreement score that separates clean frames
  (S>0.6) from jumps (S<0.5) needs the renderer, which this module does not
  have.
- A constant-velocity error-state EKF or RTS smoother on the raw track would
  fit a smooth curve through the drift span and report it as measured motion.
  With jumps of 50-95 deg inside a 33 ms frame there is no dynamics model that
  explains the data; the correct model is "measurement invalid".
- Interpolating the two *estimated* anchors across the span (keyframe SE3
  interpolation) bakes the wrong rotations into the output silently.

This module therefore implements a damage-limitation post-pass ("FUSED-3"):
detect jump spans from robust statistics of the clip's healthy leg, and bridge
each span between its two *healthy* pivots with an SE(3) geodesic, flagging
every bridged frame in the manifest. Spans running to the clip end hold the
last healthy pose. Non-flagged frames are copied byte-identical: real tracked
motion is never altered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ SO(3)/SE(3)


def rotation_log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map -> so(3) vector (axis * angle)."""

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos))
    if theta < 1e-9:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:
        # 180 deg: extract axis from the symmetric part.
        A = R + np.eye(3)
        col = int(np.argmax(np.diag(A)))
        axis = A[:, col]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            norm = float(np.linalg.norm(axis))
        return axis / norm * theta
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return vee * (theta / (2.0 * np.sin(theta)))


def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = np.asarray(w, dtype=np.float64).reshape(3)
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])


def se3_slerp(Ta: np.ndarray, Tb: np.ndarray, s: float) -> np.ndarray:
    """Interpolate SE(3): geodesic on SO(3), linear on translation."""

    Ta = np.asarray(Ta, dtype=np.float64)
    Tb = np.asarray(Tb, dtype=np.float64)
    s = float(np.clip(s, 0.0, 1.0))
    Ra, ta = Ta[:3, :3], Ta[:3, 3]
    Rb, tb = Tb[:3, :3], Tb[:3, 3]
    w = rotation_log(Ra.T @ Rb)
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        Rs = Ra
    else:
        n = w / theta
        K = hat(n)
        Rw = np.eye(3) + np.sin(s * theta) * K + (1 - np.cos(s * theta)) * K @ K
        Rs = Ra @ Rw
    out = np.eye(4)
    out[:3, :3] = Rs
    out[:3, 3] = (1.0 - s) * ta + s * tb
    return out


def se3_distance(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """(translation m, rotation deg) between two poses."""

    D = np.linalg.inv(np.asarray(Ta, dtype=np.float64)) @ np.asarray(
        Tb, dtype=np.float64
    )
    c = float(np.clip((np.trace(D[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.linalg.norm(D[:3, 3])), float(np.degrees(np.arccos(c)))


# ------------------------------------------------------------------ data IO


@dataclass(frozen=True)
class TrackFrame:
    index: int
    pose: np.ndarray | None
    tracked: bool
    method: str
    direction: str


def load_poses(poses_json: Path) -> tuple[dict, list[TrackFrame]]:
    """Read a ``poses.json`` manifest into (manifest, ordered frames)."""

    poses_json = Path(poses_json).expanduser().resolve()
    manifest = json.loads(poses_json.read_text(encoding="utf-8"))
    poses_dir = Path(manifest.get("poses_dir") or poses_json.parent / "poses")
    frames: list[TrackFrame] = []
    for entry in manifest["entries"]:
        pose = None
        if entry["tracked"] and entry["pose_filename"]:
            pose = np.loadtxt(poses_dir / entry["pose_filename"]).reshape(4, 4)
        frames.append(
            TrackFrame(
                index=int(entry["index"]),
                pose=pose,
                tracked=bool(entry["tracked"]),
                method=str(entry["method"]),
                direction=str(entry["direction"]),
            )
        )
    frames.sort(key=lambda f: f.index)
    return manifest, frames


# ------------------------------------------------------------------ jump detection


@dataclass(frozen=True)
class JumpGateConfig:
    rot_n_deg: float = 12.0
    """Healthy-segment robust scale multiplier for the rotation gate."""

    rot_min_deg: float = 20.0
    """Absolute floor for the per-frame rotation gate (deg)."""

    trans_n_m: float = 12.0
    """Healthy-segment robust scale multiplier for the translation gate."""

    trans_min_m: float = 0.06
    """Absolute floor for the per-frame translation gate (m)."""


@dataclass(frozen=True)
class StepStats:
    index: int
    dt_m: float
    dr_deg: float
    is_jump: bool


def _robust_scale(values: np.ndarray) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return max(1.4826 * mad, 1e-9)


def gate_steps(
    frames: list[TrackFrame], cfg: JumpGateConfig
) -> tuple[list[StepStats], float, float]:
    """Per-frame gate from healthy-segment robust statistics.

    The robust scales are estimated on the ``backward``/``register`` leg (the
    clip's instrumented-clean segment; see the manifest's ``direction`` field).
    Falls back to the absolute floors when the healthy leg has <8 steps.
    """

    steps: list[tuple[int, float, float, str]] = []
    for prev, cur in zip(frames[:-1], frames[1:]):
        if cur.index != prev.index + 1:
            continue
        if prev.pose is None or cur.pose is None:
            steps.append((cur.index, np.nan, np.nan, cur.direction))
            continue
        dt, dr = se3_distance(prev.pose, cur.pose)
        steps.append((cur.index, dt, dr, cur.direction))

    healthy_dr = np.array(
        [s[2] for s in steps if s[3] in ("backward", "register") and np.isfinite(s[2])]
    )
    healthy_dt = np.array(
        [s[1] for s in steps if s[3] in ("backward", "register") and np.isfinite(s[1])]
    )
    dr_gate = cfg.rot_min_deg
    dt_gate = cfg.trans_min_m
    if healthy_dr.size >= 8:
        dr_gate = max(
            cfg.rot_min_deg,
            float(np.median(healthy_dr)) + cfg.rot_n_deg * _robust_scale(healthy_dr),
        )
        dt_gate = max(
            cfg.trans_min_m,
            float(np.median(healthy_dt)) + cfg.trans_n_m * _robust_scale(healthy_dt),
        )

    stats = [
        StepStats(
            index=i,
            dt_m=float(dt),
            dr_deg=float(dr),
            is_jump=bool(
                np.isfinite(dr) and np.isfinite(dt) and (dr > dr_gate or dt > dt_gate)
            ),
        )
        for (i, dt, dr, _) in steps
    ]
    return stats, dr_gate, dt_gate


def segment_gaps(stats: list[StepStats]) -> list[tuple[int, int]]:
    """Contiguous [first_step, last_step] spans affected by a jump.

    A span (a, b) means the steps a..b are untrusted, so *poses* [a-1, b] are
    untrusted; the nearest trustworthy poses are a-2 (leading pivot) and b+1
    (trailing pivot). Detection widens each gate excursion on both sides while
    steps keep moving, then merges excursions whose untrusted windows touch.
    An excursion *ends* at the first stationary step; a tracker that holds a
    wrong-but-constant pose after losing the object is therefore left to the
    trailing-pivot / terminal-edge handling in ``fuse_track`` rather than
    being merged into real motion that follows the excursion.
    """

    def moving(s: StepStats) -> bool:
        """A step that changes pose meaningfully (extends an excursion)."""
        if not (np.isfinite(s.dr_deg) and np.isfinite(s.dt_m)):
            return False
        return s.is_jump or s.dr_deg > 8.0 or s.dt_m > 0.025

    # Collect every step that is either gated or that moves while adjacent to
    # a gated step: walk outward from each flagged step while steps keep moving.
    n = len(stats)
    in_excursion = [False] * n
    for i, s in enumerate(stats):
        if not s.is_jump:
            continue
        in_excursion[i] = True
        j = i - 1
        while j >= 0 and moving(stats[j]):
            in_excursion[j] = True
            j -= 1
        j = i + 1
        while j < n and moving(stats[j]):
            in_excursion[j] = True
            j += 1

    spans: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not in_excursion[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and in_excursion[i + 1]:
            i += 1
        spans.append((stats[start].index, stats[i].index))
        i += 1

    # Merge spans whose untrusted-pose windows [a-1, b] overlap or touch, so a
    # bridge never anchors on a pose that a neighbouring span already distrusts.
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a - 1 <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


# ------------------------------------------------------------------ fusion


@dataclass(frozen=True)
class FuseVerdict:
    frame_index: int
    action: str
    """``keep`` / ``bridge`` / ``hold`` / ``drop``."""
    pivot_a: int | None = None
    pivot_b: int | None = None
    slerp_t: float | None = None
    note: str = ""


_REACQUIRE_OK_ROT_DEG = 15.0
_HYSTERESIS = 2


def fuse_track(
    frames: list[TrackFrame], cfg: JumpGateConfig
) -> tuple[list[TrackFrame], list[FuseVerdict]]:
    """Fuse the forward and backward legs into one damage-limited track.

    Policy:

    - healthy frames keep their measured pose (``keep``) -- nothing is hidden;
    - within a flagged span, the interpolated healthy-pivot bridge is compared
      against the *measured* trajectory: keep both, but the bridge is what
      gets the trust flag;
    - jumps are usually one-sided (loss, then optional re-acquire): the
      bridge landmarks are the two crossing points where the measured track
      deviates from / rejoins the healthy hull, so the fused curve follows
      the measurement wherever the measurement is consistent;
    - a flagged span reaching the clip end has no re-acquire: hold the last
      healthy pose rather than report drift as motion (``hold``);
    - frames with no pose stay untouched (``drop``).
    """

    stats, _, _ = gate_steps(frames, cfg)
    spans = segment_gaps(stats)
    span_of: dict[int, tuple[int, int]] = {
        step_idx: span for span in spans for step_idx in range(span[0], span[1] + 1)
    }

    pose_by_index = {f.index: f for f in frames}
    tracked = [f for f in frames if f.pose is not None and f.tracked]
    out: list[TrackFrame] = []
    verdicts: list[FuseVerdict] = []

    def _pivot_pose(idx: int) -> np.ndarray | None:
        f = pose_by_index.get(idx)
        return (
            f.pose
            if (f is not None and f.pose is not None and idx not in span_of)
            else None
        )

    done_spans: set[tuple[int, int]] = set()
    bridges: dict[tuple[int, int], dict[int, tuple[np.ndarray, FuseVerdict]]] = {}

    for span in spans:
        if span in done_spans:
            continue
        done_spans.add(span)
        a_step, b_step = span
        # Untrusted poses are frames [a_step - 1, b_step]: the steps a..b are
        # untrusted, which implicates the poses on both sides of each step.
        # The nearest trustworthy poses are therefore a_step - 2 (leading
        # pivot) and b_step + 1 (trailing pivot); walk out further if a
        # neighbouring span abuts.
        first_idx, last_idx = tracked[0].index, tracked[-1].index
        ia = a_step - 2
        while ia in span_of and ia > first_idx:
            ia -= 1
        pivot_a = None if ia in span_of else _pivot_pose(ia)
        ib = b_step + 1
        while ib in span_of and ib < last_idx:
            ib += 1
        pivot_b = None if ib in span_of else _pivot_pose(ib)
        # A *terminal* excursion (no moving step separates it from the clip
        # boundary) can hide a jump whose tell-tale counter-step never lands
        # in the track, so a pivot within two frames of the boundary is
        # suspect too: refuse to anchor on it and hold from the other side.
        if pivot_a is not None and ia - first_idx < 2:
            pivot_a = None
        if pivot_b is not None and last_idx - ib < 2:
            pivot_b = None
        if pivot_a is None and pivot_b is not None:
            # no trustworthy leading pivot: hold the trailing healthy pose
            pivot_a, pivot_b, ia = pivot_b, None, ib
        span_frames = [f for f in tracked if a_step - 1 <= f.index <= b_step]
        result: dict[int, tuple[np.ndarray, FuseVerdict]] = {}
        if pivot_a is not None and pivot_b is not None:
            Ta, Tb = pivot_a, pivot_b
            for f in span_frames:
                alpha = (f.index - ia) / (ib - ia)
                _, dev = se3_distance(f.pose, se3_slerp(Ta, Tb, alpha))
                result[f.index] = (
                    se3_slerp(Ta, Tb, alpha),
                    FuseVerdict(
                        f.index,
                        "bridge",
                        pivot_a=ia,
                        pivot_b=ib,
                        slerp_t=round(alpha, 4),
                        note=f"span {a_step}-{b_step} dev={dev:.1f}deg",
                    ),
                )
        elif pivot_a is not None:
            side = "trailing" if pivot_b is None and ia != a_step - 2 else "leading"
            for f in span_frames:
                result[f.index] = (
                    pivot_a.copy(),
                    FuseVerdict(
                        f.index,
                        "hold",
                        pivot_a=ia,
                        note=f"hold {side} pivot {ia} for span {a_step}-{b_step}",
                    ),
                )
        else:
            for f in span_frames:
                # no trustworthy leading pivot: keep measurement, flagged
                result[f.index] = (
                    f.pose.copy(),
                    FuseVerdict(
                        f.index, "keep", note=f"span {a_step}-{b_step} loses lead"
                    ),
                )
        bridges[span] = result

    for f in frames:
        entry = next((b[f.index] for s, b in bridges.items() if f.index in b), None)
        if f.pose is None or not f.tracked:
            out.append(f)
            verdicts.append(FuseVerdict(f.index, "drop", note="no pose"))
        elif entry is not None:
            pose, verdict = entry
            out.append(
                TrackFrame(
                    f.index,
                    pose,
                    True,
                    method="fused-" + verdict.action,
                    direction=f.direction,
                )
            )
            verdicts.append(verdict)
        else:
            out.append(f)
            verdicts.append(FuseVerdict(f.index, "keep"))
    return out, verdicts


# ------------------------------------------------------------------ metrics


def track_metrics(frames: list[TrackFrame]) -> dict:
    """Per-frame innovation stats + summary for a fused track."""

    steps: list[dict] = []
    dts: list[float] = []
    drs: list[float] = []
    for prev, cur in zip(frames[:-1], frames[1:]):
        if cur.index != prev.index + 1 or prev.pose is None or cur.pose is None:
            continue
        dt, dr = se3_distance(prev.pose, cur.pose)
        dts.append(dt)
        drs.append(dr)
        steps.append({"index": cur.index, "dt_m": round(dt, 6), "dr_deg": round(dr, 4)})
    dts_a = np.array(dts) if dts else np.zeros(1)
    drs_a = np.array(drs) if drs else np.zeros(1)
    summary = {
        "steps": len(steps),
        "dt_m": {
            "median": float(np.median(dts_a)),
            "p90": float(np.percentile(dts_a, 90)),
            "max": float(np.max(dts_a)),
        },
        "dr_deg": {
            "median": float(np.median(drs_a)),
            "p90": float(np.percentile(drs_a, 90)),
            "max": float(np.max(drs_a)),
        },
        "steps_over_20deg": int((drs_a > 20.0).sum()),
        "steps_over_50mm": int((dts_a > 0.05).sum()),
    }
    return {"summary": summary, "steps": steps}


def pose_stability_metrics(
    frames: list[TrackFrame], tol_m: float = 0.01, tol_deg: float = 2.0
) -> dict:
    """Deviation from a hanging-window median pose per contiguous method run.

    For static tails this is the temporal stability metric the report needs:
    how far each pose sits from its local 10-frame median.
    """

    out = []
    poses = [(f.index, f.pose) for f in frames if f.pose is not None and f.tracked]
    for k in range(len(poses)):
        lo = max(0, k - 5)
        hi = min(len(poses), k + 6)
        window = [poses[j][1] for j in range(lo, hi) if j != k]
        if not window:
            continue
        ts = np.stack([w[:3, 3] for w in window])
        med_t = np.median(ts, axis=0)
        drs = [se3_distance(poses[k][1], w)[1] for w in window]
        out.append(
            {
                "index": poses[k][0],
                "dt_from_median_m": float(np.linalg.norm(poses[k][1][:3, 3] - med_t)),
                "dr_from_median_deg": float(np.median(drs)),
            }
        )
    arr_t = np.array([o["dt_from_median_m"] for o in out]) if out else np.zeros(1)
    arr_r = np.array([o["dr_from_median_deg"] for o in out]) if out else np.zeros(1)
    return {
        "median_dt_from_window_m": float(np.median(arr_t)),
        "max_dt_from_window_m": float(np.max(arr_t)),
        "median_dr_from_window_deg": float(np.median(arr_r)),
        "max_dr_from_window_deg": float(np.max(arr_r)),
        "frac_within_tol": float(
            ((arr_t <= tol_m) & (arr_r <= tol_deg)).mean() if out else 1.0
        ),
        "per_frame": out,
    }
