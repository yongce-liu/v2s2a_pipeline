"""Process v2s2a-pipeline stage outputs into the retargeting pipeline format.

Load hand (``hand_recon``) + object pose (``pose_estimation``) tracks from the
clip's stage manifests, gravity-align to Z-up, velocity-cap spike cleaning, and
write the qpos trajectory. Input layout is the v2s2a ``outputs/<clip>/``
contract; see ``scene_construction.motion_sources``.
"""

import json
import os
import shutil

import loguru
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from scene_construction.in_hand import (
    compute_in_hand_mask,
    find_freeze_indices,
    in_hand_at_endpoint,
)
from scene_construction.io import get_mesh_dir, get_processed_data_dir
from scene_construction.motion_sources import ClipInputs, load_clip_inputs

# OpenPose 21-joint indices per finger (thumb, index, middle, ring, pinky).
# Thumb has no anatomical PIP/DIP — its CMC/MCP/IP/tip are mapped onto the same
# slots: PIP-analog = MCP (idx 2), DIP-analog = IP (idx 3).
FINGERTIP_JOINT_IDX = [4, 8, 12, 16, 20]
PIP_JOINT_IDX = [2, 6, 10, 14, 18]
DIP_JOINT_IDX = [3, 7, 11, 15, 19]

# ---------------------------------------------------------------------------
# Velocity-capped spike detection + interpolation.
#
# Fixed per-signal thresholds (derived from a dataset-wide noise analysis).
# A frame-to-frame edge is flagged
# when its velocity exceeds `min(median + K_MAD * MAD, V_CAP)`.  The edge mask
# is converted to a frame mask by OR-ing across both endpoints; short gaps are
# merged; masked regions longer than MAX_BURST are left alone (presumed real
# motion); remaining regions are interpolated (linear / SLERP).  No low-pass
# smoothing — retargeting absorbs residual jitter.
# ---------------------------------------------------------------------------
CLEAN_CONFIG = {
    # name   k_mad, v_cap (m/fr or rad/fr), window (for local median+MAD)
    "pos": dict(k_mad=8.0, v_cap=0.20, window=31),
    "rot": dict(k_mad=8.0, v_cap=0.40, window=31),
}
# Post-processing of the shared mask (after OR across all signals).
SHARED_GAP_MERGE = 1
SHARED_MAX_BURST = 10


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] for True runs in a 1-D bool mask."""
    n = len(mask)
    out = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _post_process_mask(bad: np.ndarray, gap_merge: int, max_burst: int) -> np.ndarray:
    """Merge short good-runs sandwiched between bad-runs, then unmask
    bad-runs longer than ``max_burst``.  Clip boundaries act as implicit bad
    neighbors so a short good prefix/suffix adjacent to a real bad run also
    merges — otherwise the interpolation ramp leaves the wrong-valued boundary
    frames untouched and re-introduces the jump we wanted to smooth over.
    """
    out = bad.copy()
    n = len(out)
    if not out.any():
        return out
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if (j - i) <= gap_merge:
                out[i:j] = True
            i = j
        else:
            i += 1
    for s, e in _runs(out):
        if e - s > max_burst:
            out[s:e] = False
    return out


def _interp_positions(x: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Linear interp of ``x[bad]``; boundary bad frames clamp to nearest good."""
    if not bad.any():
        return x
    out = x.astype(np.float64, copy=True)
    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        return out
    bad_idx = np.where(bad)[0]
    for d in range(out.shape[-1]):
        out[bad_idx, d] = np.interp(bad_idx, good_idx, out[good_idx, d])
    return out


def _interp_rotations(
    rot: np.ndarray, bad: np.ndarray, is_quat_wxyz: bool
) -> np.ndarray:
    """SLERP masked frames; ``rot`` is (N, 3) rotvec or (N, 4) wxyz quat."""
    if not bad.any():
        return rot
    out = rot.astype(np.float64, copy=True)
    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        return out
    if len(good_idx) == 1:
        # A single good frame gives Slerp nothing to interpolate (it needs
        # >= 2 key rotations) — hold that orientation constant instead.
        out[bad] = rot[good_idx[0]]
        return out

    if is_quat_wxyz:
        R_good = Rotation.from_quat(out[good_idx][:, [1, 2, 3, 0]])
    else:
        R_good = Rotation.from_rotvec(out[good_idx])
    slerp = Slerp(good_idx, R_good)

    bad_idx = np.where(bad)[0]
    in_range = (bad_idx >= good_idx[0]) & (bad_idx <= good_idx[-1])
    if in_range.any():
        R_interp = slerp(bad_idx[in_range])
        if is_quat_wxyz:
            qxyzw = R_interp.as_quat()
            out[bad_idx[in_range]] = qxyzw[:, [3, 0, 1, 2]]
        else:
            out[bad_idx[in_range]] = R_interp.as_rotvec()
    for i in bad_idx[~in_range]:
        nearest = good_idx[0] if i < good_idx[0] else good_idx[-1]
        out[i] = rot[nearest]
    return out


def _velocity_position(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(x, axis=0), axis=-1)


def _velocity_rotvec(rv: np.ndarray) -> np.ndarray:
    """Angular magnitude between consecutive rotvec frames."""
    R_rel = Rotation.from_rotvec(rv[1:]) * Rotation.from_rotvec(rv[:-1]).inv()
    return R_rel.magnitude()


def _velocity_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Angular magnitude between consecutive wxyz quaternions (sign-aligned)."""
    q2 = q.copy()
    for i in range(1, len(q2)):
        if np.dot(q2[i], q2[i - 1]) < 0:
            q2[i] = -q2[i]
    dots = np.clip(np.sum(q2[:-1] * q2[1:], axis=-1), -1.0, 1.0)
    return 2.0 * np.arccos(np.abs(dots))


def _edge_to_frame_mask(edge_bad: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[:-1] |= edge_bad
    out[1:] |= edge_bad
    return out


def _rolling_median_mad(v: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge median and MAD over a centered window, clipped at the ends."""
    n = len(v)
    half = window // 2
    med = np.empty(n)
    mad = np.empty(n)
    for i in range(n):
        w = v[max(0, i - half) : min(n, i + half + 1)]
        m = np.median(w)
        med[i] = m
        mad[i] = np.median(np.abs(w - m))
    return med, mad


def _detect_raw_mask(
    signal: np.ndarray,
    valid: np.ndarray | None,
    cfg: dict,
    kind: str,  # "pos" | "rotvec" | "quat_wxyz"
) -> np.ndarray:
    """Iterative per-signal spike detection; returns velocity-spike frames only.

    Gap-merge / max_burst are applied later on the OR-combined shared mask.
    `valid=False` frames seed the mask (so their garbage doesn't pollute the
    velocity/MAD passes) but are excluded from the return value: the caller
    re-applies validity afterward, so max_burst never un-flags a long invalid run.
    """
    n = len(signal)
    if n < 3:
        return np.zeros(n, dtype=bool)
    seed = np.zeros(n, dtype=bool) if valid is None else ~valid.astype(bool)
    bad = seed.copy()

    vel_fn = {
        "pos": _velocity_position,
        "rotvec": _velocity_rotvec,
        "quat_wxyz": _velocity_quat_wxyz,
    }[kind]
    is_rot = kind != "pos"

    for _ in range(2):
        if bad.any():
            work = (
                _interp_rotations(signal, bad, kind == "quat_wxyz")
                if is_rot
                else _interp_positions(signal, bad)
            )
        else:
            work = signal
        v = vel_fn(work)
        med, mad = _rolling_median_mad(v, cfg["window"])
        thresh = np.minimum(med + cfg["k_mad"] * np.maximum(mad, 1e-10), cfg["v_cap"])
        vel_bad = _edge_to_frame_mask(v > thresh, n)
        new_bad = bad | vel_bad
        if np.array_equal(new_bad, bad):
            break
        bad = new_bad
    return bad & ~seed


def _log_mask(mask: np.ndarray, name: str) -> None:
    n_masked = int(mask.sum())
    if n_masked == 0:
        loguru.logger.debug(f"clean[{name}]: no frames flagged")
        return
    runs = _runs(mask)
    run_lens = [e - s for s, e in runs]
    loguru.logger.info(
        f"clean[{name}]: {n_masked} frames in {len(runs)} runs "
        f"(max {max(run_lens)}, lens {run_lens})"
    )


def compute_wrist_rotation(joints: np.ndarray, is_right: bool = True) -> Rotation:
    """Compute wrist rotation(s) from hand joint positions.

    Accepts (21, 3) or (..., 21, 3); returns a matching-shape Rotation.

    Coordinate frame:
    - z: middle MCP (joint 9) -> wrist (joint 0)
    - y_aux: index MCP (joint 5) -> ring MCP (joint 13) [right hand]
             ring MCP (joint 13) -> index MCP (joint 5) [left hand]
    - x = cross(y_aux, z), y = cross(z, x)
    """

    def _norm(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    z = _norm(joints[..., 9, :] - joints[..., 0, :])
    y_src = (
        joints[..., 5, :] - joints[..., 13, :]
        if is_right
        else joints[..., 13, :] - joints[..., 5, :]
    )
    y_aux = _norm(y_src)
    x = _norm(np.cross(y_aux, z))
    y = _norm(np.cross(z, x))
    return Rotation.from_matrix(np.stack([x, y, z], axis=-1))


def _identity_qpos(shape: tuple[int, ...]) -> np.ndarray:
    q = np.zeros(shape)
    q[..., 3] = 1.0
    return q


def _wrist_frame_offset(
    joints: np.ndarray, rotvec: np.ndarray, is_right: bool
) -> Rotation:
    """Fixed rotation mapping MANO canonical wrist frame to the MCP-based
    geometric frame used historically by compute_wrist_rotation.

    MCP joints are fixed in the MANO wrist frame (they are roots of finger
    chains, unaffected by hand_pose), so R_offset = R_mano^-1 @ R_geom is
    frame-invariant by construction. We average across frames via quaternion
    mean to denoise joint-position noise.
    """
    R_geom = compute_wrist_rotation(joints, is_right)
    R_mano = Rotation.from_rotvec(rotvec)
    q = (R_mano.inv() * R_geom).as_quat()  # (N, 4) xyzw
    q = q * np.sign(q @ q[0])[:, None]  # align signs before averaging
    _, eigvecs = np.linalg.eigh(q.T @ q)
    return Rotation.from_quat(eigvecs[:, -1])


def _build_side_qpos(
    joints: np.ndarray,
    fingertips: np.ndarray,
    rotvec: np.ndarray,
    world_offset: np.ndarray,
    is_right: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised (qpos_wrist, qpos_finger) for one hand side.

    Wrist orientation = MANO global_orient * fitted frame-offset, placing it
    in the same frame as compute_wrist_rotation (MCP-based).
    """
    N = joints.shape[0]
    wrist_pos = joints[:, 0, :] - world_offset
    R_off = _wrist_frame_offset(joints, rotvec, is_right)
    q_xyzw = (Rotation.from_rotvec(rotvec) * R_off).as_quat()
    qpos_wrist = np.concatenate([wrist_pos, q_xyzw[:, [3, 0, 1, 2]]], axis=-1)

    ft_pos = fingertips - world_offset
    ft_quat = np.broadcast_to([1.0, 0.0, 0.0, 0.0], (N, 5, 4))
    qpos_finger = np.concatenate([ft_pos, ft_quat], axis=-1)
    return qpos_wrist, qpos_finger


def _load_obj_verts(path: str) -> np.ndarray:
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(x) for x in parts[1:4]])
    return np.array(verts) if verts else np.zeros((0, 3))


def _scale_and_save_obj(src_path: str, scale: float, dst_path: str) -> None:
    out_lines = []
    with open(src_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                xyz = [float(x) * scale for x in parts[1:4]]
                extra = parts[4:]  # vertex colors if present
                new_line = "v " + " ".join(f"{v:.8f}" for v in xyz)
                if extra:
                    new_line += " " + " ".join(extra)
                out_lines.append(new_line + "\n")
            else:
                out_lines.append(line)
    with open(dst_path, "w") as f:
        f.writelines(out_lines)


def _copy_obj_texture(src_obj: str, dst_dir: str) -> str | None:
    """Copy the OBJ's ``map_Kd`` texture image (if any) into dst_dir.

    MuJoCo ignores OBJ ``mtllib``, so this image is never loaded by the
    sim/optimizer/visualizer; only the figure renderer uses it, attaching it via
    MJCF on a throwaway spec copy at snapshot time (see ``viser_viewer``). Saved as
    ``visual_texture.<ext>`` next to ``visual.obj``. Returns the basename or None.
    """
    src_dir = os.path.dirname(src_obj)
    mtl = os.path.join(src_dir, "material.mtl")
    if not os.path.exists(mtl):
        return None
    tex_name = None
    with open(mtl) as f:
        for line in f:
            if line.lower().startswith("map_kd"):
                tex_name = line.split(maxsplit=1)[1].strip()
                break
    if not tex_name or not os.path.exists(os.path.join(src_dir, tex_name)):
        return None
    dst = os.path.join(dst_dir, "visual_texture" + os.path.splitext(tex_name)[1])
    shutil.copyfile(os.path.join(src_dir, tex_name), dst)
    return os.path.basename(dst)


def main(
    clip_root: str,
    output_root_dir: str = "outputs",
    task: str | None = None,
    data_id: int = 0,
    embodiment_type: str = "auto",
    dataset_name: str = "do_as_i_do",
    object_name: str | None = None,
    gravity_up_cam: np.ndarray | None = None,
    gravity_meta: dict | None = None,
    inputs: ClipInputs | None = None,
    force: bool = False,
    start_frame: int = 0,
) -> str:
    output_root_dir = os.path.abspath(output_root_dir)
    clip_root = os.path.abspath(clip_root)

    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0; got {start_frame}")

    if inputs is None:
        if task is None:
            task = os.path.basename(clip_root)
        inputs = load_clip_inputs(
            clip_root,
            task=task,
            object_name=object_name,
            embodiment_type=embodiment_type,
            gravity_up_cam=gravity_up_cam,
            gravity_meta=gravity_meta,
        )
    task = task or inputs.task
    object_name = inputs.object_name
    embodiment_type = inputs.embodiment_type

    out_path = os.path.join(
        get_processed_data_dir(
            output_root_dir, dataset_name, "mano", embodiment_type, task, data_id
        ),
        "trajectory_keypoints.npz",
    )
    if not force and os.path.exists(out_path):
        loguru.logger.info(f"Skipping process_dataset.py (output exists: {out_path})")
        return task

    process_right = embodiment_type in ["right", "bimanual"]
    process_left = embodiment_type in ["left", "bimanual"]

    # ------------------------------------------------------------------
    # 1. Load hand data (hand_recon stage's meshes.npz; do-as-i-do key set)
    # ------------------------------------------------------------------
    npz_path = str(inputs.hand_npz_path)
    meshes = np.load(npz_path)

    if process_right:
        right_joints = meshes["right_joints"].copy()  # (N, 21, 3) in camera space
        right_valid_mask = meshes["right_valid"]  # (N,) bool
        right_rot = meshes["right_rot"].copy()  # (N, 3) MANO global_orient
        right_vertices = meshes["right_vertices"].astype(np.float64)  # (N, 778, 3)
        right_faces = meshes["right_faces"].astype(np.int32)  # (1552, 3)
        # 45-D MANO finger axis-angle (joints 1..15 in wrist-local frame).
        # Invariant under world-frame gravity alignment.
        right_hand_pose = meshes["right_hand_pose"].astype(np.float64)  # (N, 45)
        right_betas = meshes["right_betas"].astype(np.float64)  # (N, 10)
        N = right_joints.shape[0]

    if process_left:
        left_joints = meshes["left_joints"].copy()  # (N, 21, 3) in camera space
        left_valid_mask = meshes["left_valid"]  # (N,) bool
        left_rot = meshes["left_rot"].copy()  # (N, 3) MANO global_orient
        left_vertices = meshes["left_vertices"].astype(np.float64)  # (N, 778, 3)
        left_faces = meshes["left_faces"].astype(np.int32)  # (1552, 3)
        left_hand_pose = meshes["left_hand_pose"].astype(np.float64)  # (N, 45)
        left_betas = meshes["left_betas"].astype(np.float64)  # (N, 10)
        N = left_joints.shape[0]

    loguru.logger.info(f"Loaded hand data: {N} frames from {npz_path}")

    # ------------------------------------------------------------------
    # 1b. Sanity guard: a hand needs >= 2 valid frames to interpolate through.
    # ------------------------------------------------------------------
    if process_right:
        n_valid = int(np.asarray(right_valid_mask, dtype=bool).sum())
        if n_valid < 2:
            raise ValueError(
                f"Right hand unusable: only {n_valid}/{N} valid frames — "
                f"too few to interpolate."
            )
    if process_left:
        n_valid = int(np.asarray(left_valid_mask, dtype=bool).sum())
        if n_valid < 2:
            raise ValueError(
                f"Left hand unusable: only {n_valid}/{N} valid frames — "
                f"too few to interpolate."
            )

    # ------------------------------------------------------------------
    # 2. Object trajectory (pose_estimation stage, already metric camera frame)
    # ------------------------------------------------------------------
    obj_trans_cam = inputs.obj_trans_cam.copy()
    obj_quat_cam = inputs.obj_quat_cam.copy()
    obj_valid = inputs.obj_valid.copy()
    if len(obj_trans_cam) < N:
        pad = N - len(obj_trans_cam)
        obj_trans_cam = np.concatenate([obj_trans_cam, np.zeros((pad, 3))], axis=0)
        obj_quat_cam = np.concatenate(
            [obj_quat_cam, np.tile([1.0, 0, 0, 0], (pad, 1))], axis=0
        )
        obj_valid = np.concatenate([obj_valid, np.zeros(pad, dtype=bool)], axis=0)
    elif len(obj_trans_cam) > N:
        obj_trans_cam = obj_trans_cam[:N]
        obj_quat_cam = obj_quat_cam[:N]
        obj_valid = obj_valid[:N]

    # ------------------------------------------------------------------
    # 2b. Drop the first `start_frame` reference frames.
    # ------------------------------------------------------------------
    # All per-frame arrays are sliced in lockstep, so frame 0 below becomes the
    # new start: centering, the floor lift, the in-hand freeze window, and
    # speed resampling all then operate over the trimmed trajectory.
    if start_frame > 0:
        if start_frame >= N - 1:
            raise ValueError(f"start_frame={start_frame} leaves < 2 of {N} frames")
        obj_trans_cam = obj_trans_cam[start_frame:]
        obj_quat_cam = obj_quat_cam[start_frame:]
        obj_valid = obj_valid[start_frame:]
        if process_right:
            right_joints = right_joints[start_frame:]
            right_valid_mask = right_valid_mask[start_frame:]
            right_rot = right_rot[start_frame:]
            right_vertices = right_vertices[start_frame:]
            right_hand_pose = right_hand_pose[start_frame:]
            right_betas = right_betas[start_frame:]
        if process_left:
            left_joints = left_joints[start_frame:]
            left_valid_mask = left_valid_mask[start_frame:]
            left_rot = left_rot[start_frame:]
            left_vertices = left_vertices[start_frame:]
            left_hand_pose = left_hand_pose[start_frame:]
            left_betas = left_betas[start_frame:]
        N -= start_frame
        loguru.logger.info(f"start_frame={start_frame}: trimmed to {N} frames")

    # Object frames needing force-interpolation (negative-depth / missing).
    # Kept separate so the shared-mask max_burst step can't later treat a long
    # run of them as real motion (see step 4 below).
    obj_invalid = ~obj_valid
    n_obj_valid = int(obj_valid.sum())
    if n_obj_valid < 2:
        raise ValueError(
            f"Object trajectory unusable: only {n_obj_valid}/{N} frames have "
            f"an object pose — too few to interpolate."
        )

    # ------------------------------------------------------------------
    # 3. Gravity alignment — rotate all camera-frame data so that Z points up
    # ------------------------------------------------------------------
    gravity_up = np.asarray(inputs.gravity_up_cam, dtype=np.float64)
    gravity_meta = inputs.gravity_meta
    R_align, _ = Rotation.align_vectors([[0, 0, 1]], [gravity_up])
    loguru.logger.info(
        f"Gravity: up={gravity_up.round(3).tolist()}  "
        f"roll={gravity_meta.get('roll_deg', float('nan')):.2f}°  "
        f"pitch={gravity_meta.get('pitch_deg', float('nan')):.2f}°  "
        f"→ R_align magnitude={np.degrees(R_align.magnitude()):.1f}°"
    )

    # Apply R_align to all camera-frame arrays (vectorised).
    obj_trans_cam = R_align.apply(obj_trans_cam)  # (N, 3)
    obj_quat_xyzw = obj_quat_cam[:, [1, 2, 3, 0]]
    obj_quat_xyzw = (R_align * Rotation.from_quat(obj_quat_xyzw)).as_quat()
    obj_quat_cam = obj_quat_xyzw[:, [3, 0, 1, 2]]  # back to wxyz

    if process_right:
        right_joints = R_align.apply(right_joints.reshape(-1, 3)).reshape(N, 21, 3)
        right_rot = (R_align * Rotation.from_rotvec(right_rot)).as_rotvec()
        right_vertices = R_align.apply(right_vertices.reshape(-1, 3)).reshape(
            N, right_vertices.shape[1], 3
        )

    if process_left:
        left_joints = R_align.apply(left_joints.reshape(-1, 3)).reshape(N, 21, 3)
        left_rot = (R_align * Rotation.from_rotvec(left_rot)).as_rotvec()
        left_vertices = R_align.apply(left_vertices.reshape(-1, 3)).reshape(
            N, left_vertices.shape[1], 3
        )

    # ------------------------------------------------------------------
    # 4. Velocity-capped spike cleaning (shared hand/object mask).
    # ------------------------------------------------------------------
    # Per-signal detection, then OR the masks so a spike in any signal flags
    # the same frame for all of them.  Hand and object stay coherent: when
    # either tracker glitches, both get interpolated through.  Thresholds come
    # from a dataset-wide noise analysis.
    per_signal_masks = []
    per_signal_masks.append(
        (
            "obj_pos",
            _detect_raw_mask(obj_trans_cam, obj_valid, CLEAN_CONFIG["pos"], "pos"),
        )
    )
    per_signal_masks.append(
        (
            "obj_rot",
            _detect_raw_mask(obj_quat_cam, obj_valid, CLEAN_CONFIG["rot"], "quat_wxyz"),
        )
    )
    if process_right:
        per_signal_masks.append(
            (
                "right_wrist_pos",
                _detect_raw_mask(
                    right_joints[:, 0, :], right_valid_mask, CLEAN_CONFIG["pos"], "pos"
                ),
            )
        )
        per_signal_masks.append(
            (
                "right_wrist_rot",
                _detect_raw_mask(
                    right_rot, right_valid_mask, CLEAN_CONFIG["rot"], "rotvec"
                ),
            )
        )
    if process_left:
        per_signal_masks.append(
            (
                "left_wrist_pos",
                _detect_raw_mask(
                    left_joints[:, 0, :], left_valid_mask, CLEAN_CONFIG["pos"], "pos"
                ),
            )
        )
        per_signal_masks.append(
            (
                "left_wrist_rot",
                _detect_raw_mask(
                    left_rot, left_valid_mask, CLEAN_CONFIG["rot"], "rotvec"
                ),
            )
        )
    for name, m in per_signal_masks:
        _log_mask(m, name)

    # max_burst's "long run = real motion" heuristic is only valid for
    # velocity spikes, so post-process the spike masks alone...
    shared_mask = _post_process_mask(
        np.logical_or.reduce([m for _, m in per_signal_masks]),
        gap_merge=SHARED_GAP_MERGE,
        max_burst=SHARED_MAX_BURST,
    )
    # ...then OR in every tracker-flagged-invalid frame: low-confidence hand
    # tracking plus missing / negative-depth object poses. These must always be
    # interpolated; keeping them out of the post-process step above stops
    # max_burst from un-flagging a long invalid run.
    invalid_mask = obj_invalid.copy()
    if process_right:
        invalid_mask = invalid_mask | ~np.asarray(right_valid_mask, dtype=bool)
    if process_left:
        invalid_mask = invalid_mask | ~np.asarray(left_valid_mask, dtype=bool)
    shared_mask = shared_mask | invalid_mask
    _log_mask(shared_mask, "shared")

    obj_trans_cam = _interp_positions(obj_trans_cam, shared_mask)
    obj_quat_cam = _interp_rotations(obj_quat_cam, shared_mask, is_quat_wxyz=True)
    if process_right:
        for j in range(right_joints.shape[1]):
            right_joints[:, j, :] = _interp_positions(
                right_joints[:, j, :], shared_mask
            )
        for v in range(right_vertices.shape[1]):
            right_vertices[:, v, :] = _interp_positions(
                right_vertices[:, v, :], shared_mask
            )
        right_rot = _interp_rotations(right_rot, shared_mask, is_quat_wxyz=False)
        right_hand_pose = right_hand_pose.reshape(N, 15, 3)
        for j in range(15):
            right_hand_pose[:, j, :] = _interp_rotations(
                right_hand_pose[:, j, :], shared_mask, is_quat_wxyz=False
            )
        right_hand_pose = right_hand_pose.reshape(N, 45)
        right_betas = _interp_positions(right_betas, shared_mask)
        right_fingertips = right_joints[:, FINGERTIP_JOINT_IDX, :]
    if process_left:
        for j in range(left_joints.shape[1]):
            left_joints[:, j, :] = _interp_positions(left_joints[:, j, :], shared_mask)
        for v in range(left_vertices.shape[1]):
            left_vertices[:, v, :] = _interp_positions(
                left_vertices[:, v, :], shared_mask
            )
        left_rot = _interp_rotations(left_rot, shared_mask, is_quat_wxyz=False)
        left_hand_pose = left_hand_pose.reshape(N, 15, 3)
        for j in range(15):
            left_hand_pose[:, j, :] = _interp_rotations(
                left_hand_pose[:, j, :], shared_mask, is_quat_wxyz=False
            )
        left_hand_pose = left_hand_pose.reshape(N, 45)
        left_betas = _interp_positions(left_betas, shared_mask)
        left_fingertips = left_joints[:, FINGERTIP_JOINT_IDX, :]

    # ------------------------------------------------------------------
    # 5. Resolve object mesh and compute the world-frame shift
    # ------------------------------------------------------------------
    # The shift combines (a) centering the object's frame-0 xy at the origin
    # and (b) lifting so the lowest point of any geometry (object mesh + hand
    # vertices) over the *entire* trajectory sits at z=0. The floor in
    # the simulator is a plane at z=0, so any reference target dipping below
    # that becomes physically unreachable — checking only frame 0 (as the
    # original code did) lets later frames sneak under the floor.
    #
    # Gravity-aligned Z-up axes are preserved — NOT rotated into the object's
    # local frame — because MuJoCo's floor sits at Z=0 with gravity along -Z,
    # and rotating into the object frame would map horizontal object axes to Z.
    # Object mesh in the same frame the pose trajectory refers to
    # (pose_estimation's ``--mesh-path``; raw SAM3D units scaled here by the
    # depth-matched ``mesh_scale`` from motion_sources).
    mesh_src = str(inputs.mesh_obj_path)
    mesh_scale = inputs.mesh_scale
    loguru.logger.info(f"Object mesh: {mesh_src} (scale {mesh_scale:.4f} m/unit)")

    centering_offset = obj_trans_cam[0]
    verts = _load_obj_verts(mesh_src) * mesh_scale

    # Object world-frame z over all frames: for each frame i, z-component of
    # (R_i @ v) + t_i is (R_i[2, :] @ v) + t_i[2]. Vectorized over (N, V).
    obj_min_z = np.inf
    if len(verts) > 0:
        R_all = Rotation.from_quat(obj_quat_cam[:, [1, 2, 3, 0]]).as_matrix()
        z_axes = R_all[:, 2, :]  # (N, 3) — third row of each rotation
        obj_z_world = z_axes @ verts.T + obj_trans_cam[:, 2:3]  # (N, V)
        obj_min_z = float(obj_z_world.min())

    hand_min_z = np.inf
    if process_right and right_vertices.size > 0:
        hand_min_z = min(hand_min_z, float(right_vertices[..., 2].min()))
    if process_left and left_vertices.size > 0:
        hand_min_z = min(hand_min_z, float(left_vertices[..., 2].min()))

    traj_min_z = min(obj_min_z, hand_min_z)
    if not np.isfinite(traj_min_z):
        traj_min_z = float(centering_offset[2])

    world_offset = np.array(
        [float(centering_offset[0]), float(centering_offset[1]), traj_min_z]
    )
    loguru.logger.info(
        f"world_offset={world_offset.round(4)} "
        f"(centering_xy={centering_offset[:2].round(4)}, "
        f"traj_min_z={traj_min_z:.4f}, "
        f"obj_min_z={obj_min_z:.4f}, hand_min_z={hand_min_z:.4f})"
    )

    # ------------------------------------------------------------------
    # 6. Build trajectory in gravity-aligned Z-up frame, shifted by world_offset
    # ------------------------------------------------------------------
    qpos_obj = np.concatenate([obj_trans_cam - world_offset, obj_quat_cam], axis=-1)

    if process_right:
        qpos_wrist_right, qpos_finger_right = _build_side_qpos(
            right_joints, right_fingertips, right_rot, world_offset, is_right=True
        )
        qpos_pip_right = right_joints[:, PIP_JOINT_IDX, :] - world_offset
        qpos_dip_right = right_joints[:, DIP_JOINT_IDX, :] - world_offset
    else:
        qpos_wrist_right = _identity_qpos((N, 7))
        qpos_finger_right = _identity_qpos((N, 5, 7))
        qpos_pip_right = np.zeros((N, 5, 3))
        qpos_dip_right = np.zeros((N, 5, 3))

    if process_left:
        qpos_wrist_left, qpos_finger_left = _build_side_qpos(
            left_joints, left_fingertips, left_rot, world_offset, is_right=False
        )
        qpos_pip_left = left_joints[:, PIP_JOINT_IDX, :] - world_offset
        qpos_dip_left = left_joints[:, DIP_JOINT_IDX, :] - world_offset
    else:
        qpos_wrist_left = _identity_qpos((N, 7))
        qpos_finger_left = _identity_qpos((N, 5, 7))
        qpos_pip_left = np.zeros((N, 5, 3))
        qpos_dip_left = np.zeros((N, 5, 3))

    qpos_obj_right = qpos_obj.copy() if process_right else _identity_qpos((N, 7))
    qpos_obj_left = qpos_obj.copy() if process_left else _identity_qpos((N, 7))

    # Hand meshes in the same world frame as qpos (shift vertices by world_offset).
    # Unprocessed sides save empty arrays.
    if process_right:
        mano_verts_right = (right_vertices - world_offset).astype(np.float32)
        mano_faces_right = right_faces
    else:
        mano_verts_right = np.zeros((0, 0, 3), dtype=np.float32)
        mano_faces_right = np.zeros((0, 3), dtype=np.int32)
    if process_left:
        mano_verts_left = (left_vertices - world_offset).astype(np.float32)
        mano_faces_left = left_faces
    else:
        mano_verts_left = np.zeros((0, 0, 3), dtype=np.float32)
        mano_faces_left = np.zeros((0, 3), dtype=np.int32)

    # 45-D MANO finger axis-angle (joints 1..15, wrist-local frame).
    # Invariant under gravity rotation. Kept for reference/diagnostics.
    mano_finger_pose_right = (
        right_hand_pose.astype(np.float32)
        if process_right
        else np.zeros((0, 45), dtype=np.float32)
    )
    mano_finger_pose_left = (
        left_hand_pose.astype(np.float32)
        if process_left
        else np.zeros((0, 45), dtype=np.float32)
    )
    # Post-gravity-align, post-spike-clean global_orient (MANO body rotation
    # in data world frame). Kept for reference/diagnostics.
    mano_global_orient_right = (
        right_rot.astype(np.float32)
        if process_right
        else np.zeros((0, 3), dtype=np.float32)
    )
    mano_global_orient_left = (
        left_rot.astype(np.float32)
        if process_left
        else np.zeros((0, 3), dtype=np.float32)
    )
    # Per-frame MANO shape blendshape coefficients (raw from upstream predictor;
    # not modified by gravity alignment or world_offset).
    mano_betas_right = (
        right_betas.astype(np.float32)
        if process_right
        else np.zeros((0, 10), dtype=np.float32)
    )
    mano_betas_left = (
        left_betas.astype(np.float32)
        if process_left
        else np.zeros((0, 10), dtype=np.float32)
    )

    # ------------------------------------------------------------------
    # 6b. Save scaled object mesh
    # ------------------------------------------------------------------
    mesh_dir = get_mesh_dir(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        object_name=task,
    )
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_dst = f"{mesh_dir}/visual.obj"
    _scale_and_save_obj(mesh_src, mesh_scale, mesh_dst)
    loguru.logger.info(f"Saved scaled mesh → {mesh_dst}")
    # Depth-matching sanity: a ~1 m mesh means the reconstruction mesh was left
    # in normalized units (layout scale missing the metric `scale_m` factor).
    # A hand-manipulable object should be comfortably under 0.5 m per axis.
    _ext = _load_obj_verts(mesh_dst)
    _ext = _ext.max(0) - _ext.min(0) if len(_ext) else np.zeros(3)
    loguru.logger.info(f"Scaled mesh extent (m): {np.round(_ext, 3).tolist()}")
    if np.max(_ext) > 0.5:
        raise ValueError(
            f"Scaled object mesh is implausibly large ({np.max(_ext):.2f} m along "
            "its longest axis; a hand-manipulable object should be < 0.5 m). "
            "The reconstruction layout scale is missing the metric depth-matching "
            "factor (track_object_foundationpose.py's scale_m). Re-run Stages "
            "3-4 of the reconstruction pipeline."
        )
    tex_name = _copy_obj_texture(mesh_src, mesh_dir)
    if tex_name:
        loguru.logger.info(f"Copied object texture → {mesh_dir}/{tex_name}")

    # ------------------------------------------------------------------
    # 6.5. Freeze object reference outside the in-hand window
    # ------------------------------------------------------------------
    # For each side+endpoint where the in-hand endpoint check fails (i.e.
    # the post-IK pedestal step will add a stabilizing pedestal there),
    # find the earliest/latest frame that passes the per-frame in-hand
    # mask and clamp the surrounding frames to that pose. Holds the
    # object static during the approach (and release) so the pedestal
    # placed under it (post-IK, in ``retargeting/pipeline/resolve_pedestal.py``) matches a
    # stable reference position.
    #
    # Uses the raw visual mesh (``verts``, mesh-local) here rather than
    # the convex-decomp output, which doesn't exist yet at this stage.
    # The in-hand mask is robust to mesh source at the 10 cm threshold.
    for side, hand_verts_side, qpos_obj_side in (
        ("right", mano_verts_right, qpos_obj_right),
        ("left", mano_verts_left, qpos_obj_left),
    ):
        if hand_verts_side.shape[0] == 0 or len(verts) == 0:
            continue
        needs = {}
        for ep_name, frame_idx in (("start", 0), ("end", -1)):
            in_hand, *_ = in_hand_at_endpoint(
                hand_verts_world=hand_verts_side,
                qpos_obj=qpos_obj_side,
                obj_verts=verts,
                frame=frame_idx,
            )
            needs[ep_name] = not in_hand
        if not (needs["start"] or needs["end"]):
            continue
        mask = compute_in_hand_mask(hand_verts_side, qpos_obj_side, verts)
        i_start, j_end = find_freeze_indices(mask)
        if i_start is None:
            loguru.logger.warning(
                f"{side}: no frame passes the in-hand check; skipping reference freeze"
            )
            continue
        if needs["start"] and i_start > 0:
            qpos_obj_side[:i_start] = qpos_obj_side[i_start]
            loguru.logger.info(
                f"{side}: froze qpos_obj[0:{i_start}] = qpos_obj[{i_start}]"
            )
        if needs["end"] and j_end < qpos_obj_side.shape[0] - 1:
            qpos_obj_side[j_end + 1 :] = qpos_obj_side[j_end]
            loguru.logger.info(
                f"{side}: froze qpos_obj[{j_end + 1}:] = qpos_obj[{j_end}]"
            )

    # ------------------------------------------------------------------
    # 7. Save trajectory_keypoints.npz
    # ------------------------------------------------------------------
    out_data_dir = get_processed_data_dir(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(out_data_dir, exist_ok=True)

    np.savez(
        f"{out_data_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_pip_right=qpos_pip_right,
        qpos_dip_right=qpos_dip_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_pip_left=qpos_pip_left,
        qpos_dip_left=qpos_dip_left,
        qpos_obj_left=qpos_obj_left,
        contact_right=np.zeros((N, 10)),
        contact_pos_right=np.zeros((10, 3)),
        contact_left=np.zeros((N, 10)),
        contact_pos_left=np.zeros((10, 3)),
        centering_offset=centering_offset,
        mano_verts_right=mano_verts_right,
        mano_faces_right=mano_faces_right,
        mano_verts_left=mano_verts_left,
        mano_faces_left=mano_faces_left,
        mano_finger_pose_right=mano_finger_pose_right,
        mano_finger_pose_left=mano_finger_pose_left,
        mano_global_orient_right=mano_global_orient_right,
        mano_global_orient_left=mano_global_orient_left,
        mano_betas_right=mano_betas_right,
        mano_betas_left=mano_betas_left,
    )
    loguru.logger.info(f"Saved trajectory_keypoints.npz → {out_data_dir}")

    # task_info.json lives one level above the data_id directory
    task_dir = os.path.dirname(out_data_dir)
    mesh_dir_relative = os.path.relpath(mesh_dir, output_root_dir)
    task_info = {
        "task": task,
        "dataset_name": dataset_name,
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        # decompose_mesh.py prepends output_root_dir, so store a relative path.
        # For bimanual with a single shared object, only right_object_mesh_dir
        # is set (left_object_mesh_dir=None signals a shared object).
        "right_object_mesh_dir": mesh_dir_relative if process_right else None,
        "left_object_mesh_dir": mesh_dir_relative
        if (process_left and not process_right)
        else None,
    }
    task_info_path = f"{task_dir}/task_info.json"
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info.json → {task_info_path}")

    return task
