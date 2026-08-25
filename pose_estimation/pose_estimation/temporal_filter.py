"""Late-frame temporal filtering of object poses.

Runs *after* FoundationPose has produced every per-frame pose, so the full
trajectory (including the bidirectional passes) is available. The filter is a
constant-velocity error-state EKF over ``(translation, velocity, rotation)``
with an SO(3) attitude error, followed by a velocity-only RTS backward pass.
No keyframe interpolation is used anywhere: rejected measurements simply make
the filter coast on the process model, and accepted ones are fused with the
standard KF update.

Quality awareness
-----------------
- Measurement noise is per-frame and multiplicative on top of a base value:
  registration seeds (``register`` / ``obj-recon-seed-refine``) get a lower
  base noise; tracked frames scale with mask area, valid-depth fraction, and
  the FoundationPose refine-iteration budget.
- Innovation gating: the combined 6-DOF innovation Mahalanobis distance is
  compared against a chi2(6) threshold. A larger threshold applies to the
  first measurement after a coast run so legitimate fast motion re-acquires
  cleanly without locking onto one recovery observation.

yellow_spoon calibration (FoundationPose MV, 89 frames @ 30 fps):
- tracked translation step   : median 0.009 m, p99 0.13 m, max 0.17 m
  (real motion reaches ~0.9 m/s p90 / 5 m/s max around the flip at frames 56-60)
- tracked rotation step      : median 4 deg, p90 28 deg, p99 83 deg, max 96 deg
  (outliers happen when the symmetric bowl-of-spoon silhouette swaps ends)
- object mask area           : mean ~9k px, p05 ~2.8k px, min ~1.1k px
- metric object extent       : ~0.15 m

The defaults below therefore keep genuine 0.5 m/s translations and 800 deg/s
rotations ungated while rejecting the ~90 deg/frame symmetric flips.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# chi2 with 6 dof, 0.99 quantile.
MAHALANOBIS_6D_P99 = 16.81


# ---------------------------------------------------------------------------
# Tunables.
# ---------------------------------------------------------------------------


@dataclass
class TemporalFilterArgs:
    """Settings for the late-frame pose temporal filter."""

    enabled: bool = False
    """Enable the late-frame smoother over all produced per-frame poses."""

    dt_sec: float | None = None
    """Time step between frames; ``None`` reads ``fps`` from the run manifest."""

    sigma_v_mps: float = 0.6
    """Process-noise std of the white acceleration driving velocity
    (m/s per sqrt(s)). Sized so legitimate ~1 m/s yellow_spoon moves stay
    within ~3 sigma over a 33 ms step."""

    sigma_omega_extent_per_s: float = 1.2
    """Process-noise std of the white angular acceleration driving the
    orientation error, in rad/s per sqrt(s) per metre of object extent.
    1.2 x 0.15 m = 0.18 rad ~ 10 deg; over 33 ms that tolerates the real
    ~800 deg/s rotations while gating the multi-thousand deg/s flips."""

    meas_translation_m: float = 0.02
    """Base measurement std (m) of a *tracked* pose translation. Yellow_spoon's
    tracked translation jitter is ~0.01 m median; 0.02 m keeps real moves
    inside the ~99%% chi2 gate while rejecting the 90 deg flip translations."""

    meas_rotation_rad: float = 0.15
    """Base measurement std (rad, axis-angle) of a tracked pose rotation
    (~8.6 deg). Keeps genuine 30 deg updates inside the gate and rejects the
    >80 deg symmetric-silhouette flips."""

    registration_quality_scale: float = 0.5
    """Multiplicative noise scale for registration seeds, which are more
    trustworthy than tracked poses."""

    registration_freeze: bool = True
    """When True, registration seeds are treated as authoritative and replace
    the filter state exactly (no EKF correction). This preserves the obj_recon
    metric seed as the trajectory anchor."""

    refine_iter_reference: int = 10
    """Track refine-iteration count at which the base noise is calibrated."""

    refine_iter_noise_exponent: float = -0.5
    """Noise scales as ``(track_refine_iter / refine_iter_reference) ** exponent``;
    halving the refiner budget inflates noise by sqrt(2)."""

    mask_area_reference_px: float = 9000.0
    """Mask area (px) at which the mask-quality multiplier is 1 (yellow_spoon
    mean object-mask area)."""

    mask_area_min_quality: float = 0.25
    """Clamp on the mask-quality multiplier: small/partial masks get at most
    4x noise."""

    depth_valid_reference: float = 0.9
    """Valid-depth fraction at which the depth-quality multiplier is 1."""

    depth_valid_min_quality: float = 0.25
    """Clamp on the depth-quality multiplier."""

    gate_chi2_threshold: float = MAHALANOBIS_6D_P99
    """Innovation gate on the combined 6-DOF Mahalanobis distance (chi2, 6 dof)."""

    gate_chi2_threshold_on_coast: float = 30.0
    """More permissive gate applied to the first measurement after a coast
    run, so the filter re-acquires after legitimate fast motion."""

    write_manifest: bool = True
    """Write ``poses_filtered.json`` + ``poses_filtered/`` when True; when
    False, overwrite ``poses/`` + ``poses.json`` in place."""


# ---------------------------------------------------------------------------
# Diagnostics record written into poses_filtered.json.
# ---------------------------------------------------------------------------


@dataclass
class FilterFrameStats:
    index: int
    gated: bool = False
    coasted: bool = False
    mahalanobis: float | None = None
    mask_area_px: int | None = None
    depth_valid_fraction: float | None = None
    noise_scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Small SO(3) helpers (pure numpy, no scipy).
# ---------------------------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Log map: rotation matrix -> axis-angle vector in R^3."""

    R = np.asarray(R, dtype=np.float64)
    cos = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cos))
    if angle < 1e-12:
        return np.zeros(3)
    if abs(np.pi - angle) < 1e-6:
        # Near 180 deg: recover the axis from the symmetric part.
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
        axis[0] = np.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = np.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = np.copysign(axis[2], R[1, 0] - R[0, 1])
        n = float(np.linalg.norm(axis))
        return np.zeros(3) if n < 1e-12 else angle * axis / n
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
        2.0 * np.sin(angle)
    )
    return angle * axis


def rotation_exp(rv: np.ndarray) -> np.ndarray:
    """Exp map: axis-angle vector -> rotation matrix (Rodrigues)."""

    rv = np.asarray(rv, dtype=np.float64)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3)
    k = _skew(rv / angle)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


# ---------------------------------------------------------------------------
# Error-state EKF.
# ---------------------------------------------------------------------------

_DIM = 9  # [dp(0:3), dv(3:6), dtheta(6:9)]


@dataclass
class _KFState:
    p: np.ndarray  # translation (3,)
    v: np.ndarray  # velocity (3,)
    R: np.ndarray  # orientation (3, 3)
    P: np.ndarray  # error covariance (9, 9)

    @classmethod
    def from_pose(
        cls,
        pose: np.ndarray,
        p0_translation: float,
        p0_rotation: float,
        p0_velocity: float,
    ) -> _KFState:
        P = np.diag(
            [p0_translation**2] * 3 + [p0_velocity**2] * 3 + [p0_rotation**2] * 3
        )
        return cls(
            p=np.asarray(pose[:3, 3], dtype=np.float64).copy(),
            v=np.zeros(3),
            R=np.asarray(pose[:3, :3], dtype=np.float64).copy(),
            P=P,
        )

    def copy(self) -> _KFState:
        return _KFState(
            p=self.p.copy(), v=self.v.copy(), R=self.R.copy(), P=self.P.copy()
        )

    def to_pose(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.p
        return T


@dataclass
class _ProcessNoise:
    q_pos: float
    q_vel: float
    q_rot: float


class PoseTemporalFilter:
    """Constant-velocity error-state EKF + velocity-only RTS smoother."""

    def __init__(self, args: TemporalFilterArgs, mesh_extent_m: float = 0.15) -> None:
        self.args = args
        self.mesh_extent_m = float(mesh_extent_m)

    def _process_noise(self, dt: float) -> _ProcessNoise:
        a = self.args
        q_acc = a.sigma_v_mps**2
        q_ang = (a.sigma_omega_extent_per_s * self.mesh_extent_m) ** 2
        return _ProcessNoise(
            q_pos=q_acc * dt**3 / 3.0,
            q_vel=q_acc * dt,
            q_rot=q_ang * dt,
        )

    def _propagate(self, state: _KFState, dt: float, q: _ProcessNoise) -> None:
        F = np.eye(_DIM)
        F[0:3, 3:6] = dt * np.eye(3)
        state.P = F @ state.P @ F.T
        state.P[0:3, 0:3] += q.q_pos * np.eye(3)
        state.P[0:3, 3:6] += 0.5 * q.q_vel * dt * np.eye(3)
        state.P[3:6, 0:3] += 0.5 * q.q_vel * dt * np.eye(3)
        state.P[3:6, 3:6] += q.q_vel * np.eye(3)
        state.P[6:9, 6:9] += q.q_rot * np.eye(3)
        state.p = state.p + state.v * dt
        # Nominal heading: no explicit gyro state, so the heading stays put;
        # q_rot above covers the un-modelled inter-frame turn.

    def _measurement_R(self, scale: float) -> np.ndarray:
        a = self.args
        return (
            np.diag(
                [a.meas_translation_m * scale] * 3 + [a.meas_rotation_rad * scale] * 3
            )
            ** 2
        )

    def _update(
        self, state: _KFState, meas_pose: np.ndarray, R_mat: np.ndarray, gate: float
    ) -> tuple[bool, float]:
        meas_p = np.asarray(meas_pose[:3, 3], dtype=np.float64)
        meas_R = np.asarray(meas_pose[:3, :3], dtype=np.float64)

        innov = np.zeros(6)
        innov[0:3] = meas_p - state.p
        innov[3:6] = rotation_log(state.R.T @ meas_R)

        H = np.zeros((6, _DIM))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)

        S = H @ state.P @ H.T + R_mat
        maha = float(innov @ np.linalg.solve(S, innov))
        if not np.isfinite(maha) or maha > gate:
            return False, maha

        K = np.linalg.solve(S.T, (state.P @ H.T).T).T
        dx = K @ innov
        state.p = state.p + dx[0:3]
        state.v = state.v + dx[3:6]
        state.R = state.R @ rotation_exp(dx[6:9])
        I_KH = np.eye(_DIM) - K @ H
        state.P = I_KH @ state.P @ I_KH.T + K @ R_mat @ K.T
        state.P = 0.5 * (state.P + state.P.T)
        return True, maha

    def _rts_velocity_pass(self, filtered: list[_KFState | None], dt: float) -> None:
        """Backward pass that only re-estimates velocity.

        Positions / rotations stay exactly where the forward pass left them,
        so the anchor pose and every accepted measurement pose are preserved;
        only the velocity loses its forward-only ramp-up asymmetry.
        """

        for k in range(len(filtered) - 2, -1, -1):
            cur = filtered[k]
            nxt = filtered[k + 1]
            if cur is None or nxt is None:
                continue
            dp = nxt.p - cur.p
            v_fit = dp / max(dt, 1e-9)
            P_cur = cur.P[3:6, 3:6]
            P_nxt = nxt.P[3:6, 3:6]
            try:
                A = np.linalg.inv(P_cur)
                B = np.linalg.inv(P_nxt)
            except np.linalg.LinAlgError:
                continue
            C = 2.0 / max(float(np.trace(P_cur) + np.trace(P_nxt)), 1e-12) * np.eye(3)
            M = A + B + C
            rhs = A @ cur.v + B @ nxt.v + C @ v_fit
            try:
                cur.v = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue

    def run(
        self,
        poses: Sequence[np.ndarray | None],
        noise_scales: Sequence[float],
        dt: float,
        registration_flags: Sequence[bool] | None = None,
    ) -> tuple[list[np.ndarray | None], list[FilterFrameStats]]:
        """Filter a sequence of 4x4 poses (None = missing / untracked frame).

        ``registration_flags`` marks frames whose measurement replaces the
        filter state exactly when ``args.registration_freeze`` is set.
        """

        if len(poses) != len(noise_scales):
            raise ValueError("poses and noise_scales must be the same length")
        if registration_flags is None:
            registration_flags = [False] * len(poses)

        a = self.args
        q = self._process_noise(dt)
        filtered: list[_KFState | None] = [None] * len(poses)
        stats: list[FilterFrameStats] = []
        state: _KFState | None = None
        coast_run = 0

        for i, (pose, scale) in enumerate(zip(poses, noise_scales)):
            if pose is None:
                if state is not None:
                    self._propagate(state, dt, q)
                    coast_run += 1
                    filtered[i] = state.copy()
                    stats.append(FilterFrameStats(i, gated=False, coasted=True))
                else:
                    stats.append(FilterFrameStats(i))
                continue

            if state is None or (registration_flags[i] and a.registration_freeze):
                # Bootstrap, or snap the state to an authoritative registration
                # seed (position/rotation exact, velocity kept from the model).
                v = np.zeros(3) if state is None else state.v.copy()
                state = _KFState.from_pose(
                    pose,
                    p0_translation=a.meas_translation_m,
                    p0_rotation=max(a.meas_rotation_rad, 1e-3),
                    p0_velocity=max(a.sigma_v_mps, 1e-3),
                )
                state.v = v
                coast_run = 0
                filtered[i] = state.copy()
                stats.append(
                    FilterFrameStats(i, mahalanobis=0.0, noise_scale=float(scale))
                )
                continue

            self._propagate(state, dt, q)
            R_mat = self._measurement_R(scale)
            gate = (
                a.gate_chi2_threshold_on_coast
                if coast_run > 0
                else a.gate_chi2_threshold
            )
            accepted, maha = self._update(state, pose, R_mat, gate)
            if accepted:
                coast_run = 0
                filtered[i] = state.copy()
                stats.append(
                    FilterFrameStats(
                        i,
                        gated=False,
                        coasted=False,
                        mahalanobis=maha,
                        noise_scale=float(scale),
                    )
                )
            else:
                coast_run += 1
                filtered[i] = state.copy()
                stats.append(
                    FilterFrameStats(
                        i,
                        gated=True,
                        coasted=True,
                        mahalanobis=maha,
                        noise_scale=float(scale),
                    )
                )

        self._rts_velocity_pass(filtered, dt)
        smoothed = [None if s is None else s.to_pose() for s in filtered]
        return smoothed, stats


# ---------------------------------------------------------------------------
# Quality-aware noise scaling.
# ---------------------------------------------------------------------------


def measurement_noise_scale(
    args: TemporalFilterArgs,
    *,
    is_registration: bool,
    mask_area_px: int | None,
    depth_valid_fraction: float | None,
    track_refine_iter: int | None,
) -> float:
    """Multiplicative measurement-noise scale for one frame."""

    scale = 1.0
    if is_registration:
        scale *= args.registration_quality_scale
    if track_refine_iter is not None and track_refine_iter > 0:
        scale *= (
            track_refine_iter / float(args.refine_iter_reference)
        ) ** args.refine_iter_noise_exponent
    if mask_area_px is not None and mask_area_px > 0:
        quality = args.mask_area_reference_px / float(mask_area_px)
        scale *= min(1.0 / args.mask_area_min_quality, max(1.0, quality))
    if depth_valid_fraction is not None and depth_valid_fraction > 0.0:
        quality = args.depth_valid_reference / depth_valid_fraction
        scale *= min(1.0 / args.depth_valid_min_quality, max(1.0, quality))
    return float(scale)


# ---------------------------------------------------------------------------
# Manifest-driven entry point (late-frame pass over an existing run's outputs).
# ---------------------------------------------------------------------------


def apply_temporal_filter_to_run(
    stage_dir: Path,
    args: TemporalFilterArgs,
    *,
    mesh_extent_m: float = 0.15,
    track_refine_iter: int | None = None,
    masks_manifest: dict | None = None,
    geometry_manifest: dict | None = None,
    prompt_id: str | None = None,
) -> Path:
    """Read ``<stage_dir>/poses.json`` + ``poses/*.txt`` and write the filtered set.

    Returns the path of the manifest that was written
    (``poses_filtered.json`` by default, ``poses.json`` when overwriting).
    """

    stage_dir = Path(stage_dir)
    poses_json_path = stage_dir / "poses.json"
    if not poses_json_path.exists():
        raise FileNotFoundError(f"poses.json not found: {poses_json_path}")
    manifest = json.loads(poses_json_path.read_text(encoding="utf-8"))

    fps = float(manifest.get("fps") or 30.0)
    dt = args.dt_sec if args.dt_sec is not None else 1.0 / fps

    entries = manifest.get("entries", [])
    poses: list[np.ndarray | None] = []
    noise_scales: list[float] = []
    registration_flags: list[bool] = []
    quality_meta: list[dict] = []

    for entry in entries:
        pose = None
        pose_filename = entry.get("pose_filename")
        if entry.get("tracked") and pose_filename:
            candidate = stage_dir / "poses" / pose_filename
            if candidate.exists():
                pose = np.loadtxt(candidate).reshape(4, 4)
        poses.append(pose)

        method = entry.get("method", "")
        is_registration = method in (
            "register",
            "register-anchor",
            "obj-recon-seed-refine",
        )
        registration_flags.append(is_registration)
        mask_area, depth_frac = _quality_signals(
            int(entry.get("index", -1)),
            masks_manifest=masks_manifest,
            geometry_manifest=geometry_manifest,
            prompt_id=prompt_id,
        )
        noise_scales.append(
            measurement_noise_scale(
                args,
                is_registration=is_registration,
                mask_area_px=mask_area,
                depth_valid_fraction=depth_frac,
                track_refine_iter=track_refine_iter,
            )
        )
        quality_meta.append(
            {"mask_area_px": mask_area, "depth_valid_fraction": depth_frac}
        )

    filt = PoseTemporalFilter(args, mesh_extent_m=mesh_extent_m)
    smoothed, stats = filt.run(poses, noise_scales, dt, registration_flags)
    for s, meta in zip(stats, quality_meta):
        s.mask_area_px = meta["mask_area_px"]
        s.depth_valid_fraction = meta["depth_valid_fraction"]

    out_dir = stage_dir / ("poses_filtered" if args.write_manifest else "poses")
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry, pose in zip(entries, smoothed):
        if pose is None:
            continue
        pose_filename = entry.get("pose_filename") or f"{int(entry['index']):06d}.txt"
        np.savetxt(out_dir / pose_filename, pose, fmt="%.10f")

    filtered_manifest = dict(manifest)
    filtered_manifest["stage"] = "pose_estimation.temporal_filter"
    if args.write_manifest:
        filtered_manifest["source_poses_json"] = str(poses_json_path.resolve())
        filtered_manifest["poses_dir"] = str(out_dir.resolve())
    filtered_manifest["temporal_filter"] = {
        "dt_sec": dt,
        "mesh_extent_m": mesh_extent_m,
        "args": asdict(args),
        "gated_count": sum(1 for s in stats if s.gated),
        "coast_frame_count": sum(1 for s in stats if s.coasted),
        "frame_stats": [s.to_dict() for s in stats],
    }
    out_manifest = (
        stage_dir / "poses_filtered.json" if args.write_manifest else poses_json_path
    )
    out_manifest.write_text(
        json.dumps(filtered_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return out_manifest


def _quality_signals(
    index: int,
    *,
    masks_manifest: dict | None,
    geometry_manifest: dict | None,
    prompt_id: str | None,
) -> tuple[int | None, float | None]:
    """Per-frame (mask_area_px, depth_valid_fraction) best-effort lookup."""

    mask_area: int | None = None
    depth_frac: float | None = None

    if masks_manifest is not None:
        for m_entry in masks_manifest.get("entries", []):
            if int(m_entry.get("index", -1)) != index:
                continue
            candidates = m_entry.get("prompt_masks") or [m_entry]
            for pm in candidates:
                if prompt_id is not None and pm.get("prompt_id") != prompt_id:
                    continue
                if "area" in pm:
                    mask_area = int(pm["area"])
                break
            break

    if geometry_manifest is not None:
        for g_entry in geometry_manifest.get("entries", []):
            if int(g_entry.get("index", -1)) != index:
                continue
            frame_dir = g_entry.get("frame_dir")
            if frame_dir:
                try:
                    points = np.load(Path(frame_dir) / "points.npy", mmap_mode="r")
                    valid = np.isfinite(points[..., 2]) & (points[..., 2] > 0.001)
                    depth_frac = float(valid.mean())
                except (OSError, ValueError, KeyError):
                    depth_frac = None
            break

    return mask_area, depth_frac
