"""Pure numerical operations for hand-object alignment.

Two kinds of corrections live here:

* :func:`correction_matrix` / :func:`apply_camera_correction` — the *manual*
  override, kept for reproducibility and for power users who already know the
  exact 6 DoF. It never looks at hand data: one global rigid transform is
  left-composed onto every tracked pose. That cannot fix per-frame
  FoundationPose/HaWoR drift, depth bias that varies with object motion, or
  grip-contact errors that only exist while fingers wrap the object — which is
  why the automatic mode below exists.

* :func:`fit_pose_corrections` — the *automatic* mode, a small per-frame
  constrained re-registration inspired by do-as-i-do's physics warmup. Its
  ``warmup_min_clearance`` seeks the first penetration-free pose at contact
  distance, and its ``contact_dist_gate`` accepts as soon as the closest
  hand-object pair distance *converges* (a distance gate, not an iteration
  budget). The analogue here, without a simulator:

  1. A warmup seed — a translation-only grid search — moves the object until
     its mean Huber-clipped distance to the hand stops improving.
  2. A 6-DoF Powell polish minimizes a *contact* objective: only object
     vertices already within ``contact_band_m`` of the hand are pulled closer
     (so the fit cannot drift toward whichever hand surface happens to be
     nearest), with a hard L2 penalty on object verts inside the hand convex
     hull (the penetration term) and a strong prior keeping the correction
     inside its trust region (|t| <= ``max_translation_m``, |rot| <=
     ``max_rotation_deg``). The prior is what makes a global manual override
     unnecessary: the optimizer proposes, the gate disposes.

  Per frame we record the pre/post hand-object clearance and penetration so
  the workflow can accept or reject on measurable gates, not on vibes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def correction_matrix(
    translation_xyz: tuple[float, float, float],
    rotation_rotvec: tuple[float, float, float],
) -> np.ndarray:
    """Return a camera-frame rigid correction from metres and radians."""

    translation = np.asarray(translation_xyz, dtype=np.float64)
    rotvec = np.asarray(rotation_rotvec, dtype=np.float64)
    if translation.shape != (3,) or rotvec.shape != (3,):
        raise ValueError(
            "translation and rotation corrections must each contain 3 values"
        )
    if not np.isfinite(translation).all() or not np.isfinite(rotvec).all():
        raise ValueError("correction values must be finite")
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    correction[:3, 3] = translation
    return correction


def apply_camera_correction(pose: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Left-compose a correction so both transforms remain object-to-camera."""

    pose = np.asarray(pose, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if pose.shape != (4, 4) or correction.shape != (4, 4):
        raise ValueError("pose and correction must be 4x4 matrices")
    return correction @ pose


@dataclass(frozen=True)
class FrameFit:
    """One tracked frame's fitted correction and its acceptance metrics."""

    frame_index: int
    pre_min_dist_m: float
    pre_penetration_depth_m: float
    post_min_dist_m: float
    post_penetration_depth_m: float
    translation_xyz: tuple[float, float, float]
    rotation_rotvec: tuple[float, float, float]
    clamped: bool
    converged: bool
    objective: float


@dataclass(frozen=True)
class FitResult:
    """Aggregate of the automatic fit over every registered frame."""

    mode: str  # "per_frame" | "global"
    frames: tuple[FrameFit, ...]
    stats: dict


def correction_matrices(fit: FitResult) -> list[np.ndarray]:
    """Per-frame 4x4 corrections, ordered to match ``fit.frames``."""

    return [
        correction_matrix(frame.translation_xyz, frame.rotation_rotvec)
        for frame in fit.frames
    ]


def _subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Deterministically subsample a (M, 3) point set to at most ``max_points``."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return np.asarray(points[idx], dtype=np.float64)


def _inside_hull(hull: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Vectorized inside-test against a convex hull via plane normals.

    Trimesh's own ``contains`` needs the optional R-tree package for its broad
    phase; this convex-hull plane-dot-products version uses only numpy/scipy
    so the penetration gate runs in minimal environments. ``hull`` must be
    convex (built from :func:`trimesh.PointCloud.convex_hull`), and trimesh
    guarantees outward-facing face normals.
    """

    faces = np.asarray(hull.faces, dtype=np.int64)
    verts = np.asarray(hull.vertices, dtype=np.float64)
    centroids = verts[faces].mean(axis=1)  # (F, 3)
    normals = np.asarray(hull.face_normals, dtype=np.float64)  # (F, 3)
    offsets = np.asarray(points, dtype=np.float64)[None, :, :] - centroids[:, None, :]
    signed = np.einsum("fnc,fc->fn", offsets, normals)
    return (signed <= 1e-12).all(axis=0)


def _clearance(
    hand_verts: np.ndarray,
    object_verts_cam: np.ndarray,
    hand_hull: trimesh.Trimesh | None,
) -> tuple[float, float]:
    """(min_dist, penetration_depth) between hand and camera-frame object verts.

    ``min_dist`` is the smallest hand↔object vertex distance. ``penetration``
    is the mean hand-surface distance over object verts found *inside* the
    hand's convex hull, else 0. Negative-clearance via a real triangle mesh is
    what do-as-i-do's warmup uses; the hull approximation only engages when a
    mesh is available.
    """

    hand_verts = np.asarray(hand_verts, dtype=np.float64)
    object_verts_cam = np.asarray(object_verts_cam, dtype=np.float64)
    hand_tree = cKDTree(hand_verts)
    obj_tree = cKDTree(object_verts_cam)
    d_o2h = obj_tree.query(hand_verts, k=1)[0]
    d_h2o = hand_tree.query(object_verts_cam, k=1)[0]
    min_dist = float(min(d_o2h.min(), d_h2o.min()))
    penetration = 0.0
    if hand_hull is not None:
        inside = _inside_hull(hand_hull, object_verts_cam)
        if inside.any():
            penetration = float(d_h2o[inside].mean())
    return min_dist, penetration


def _huber(dists: np.ndarray, delta: float) -> np.ndarray:
    clipped = np.asarray(dists, dtype=np.float64)
    quad = 0.5 * clipped**2
    lin = delta * (clipped - 0.5 * delta)
    return np.where(clipped <= delta, quad, lin)


def _contact_objective(
    params: np.ndarray,
    object_verts_local: np.ndarray,
    source_pose: np.ndarray,
    hand_tree: cKDTree,
    contact_obj_idx: np.ndarray,
    contact_hand_idx: np.ndarray,
    hand_hull: trimesh.Trimesh | None,
    huber_delta: float,
    w_contact: float,
    w_penetration: float,
    w_prior: float,
    max_translation_m: float,
    max_rotation_rad: float,
) -> float:
    """Regularized contact objective.

    * Contact term: Huber mean-squared distance from the object verts inside
      the contact band (frozen correspondence from the pre-fit seed) to their
      matched hand verts. Freezing the correspondence is the do-as-i-do
      contact-gate analogue: we refine *around contact*, we do not let the
      fit wander toward whichever hand surface happens to be nearest.
    * Penetration term: L2 on object verts inside the hand hull.
    * Prior: quadratic pull toward the identity correction, in trust-region
      units. This is what keeps a global manual override unnecessary — the
      fit can only move what the evidence demands, and only this far.
    """

    if not np.isfinite(params).all():
        return float("inf")
    translation = params[:3]
    rotvec = params[3:]
    prior = (np.linalg.norm(translation) / max_translation_m) ** 2 + (
        np.linalg.norm(rotvec) / max_rotation_rad
    ) ** 2

    correction = correction_matrix(tuple(translation), tuple(rotvec))
    pose = correction @ source_pose
    obj_cam = object_verts_local @ pose[:3, :3].T + pose[:3, 3]

    if len(contact_obj_idx):
        deltas = obj_cam[contact_obj_idx] - np.asarray(
            hand_tree.data[contact_hand_idx], dtype=np.float64
        )
        contact = float(_huber(np.linalg.norm(deltas, axis=1), huber_delta).mean())
    else:
        contact = 0.0

    penetration = 0.0
    if hand_hull is not None and w_penetration > 0.0:
        inside = _inside_hull(hand_hull, obj_cam)
        if inside.any():
            depth = float(hand_tree.query(obj_cam[inside], k=1)[0].mean())
            penetration = w_penetration * depth**2

    return w_contact * contact + penetration + w_prior * prior


def _fit_frame(
    source_pose: np.ndarray,
    hand_verts: np.ndarray,
    frame_index: int,
    object_verts_local: np.ndarray,
    object_mesh: trimesh.Trimesh | None,
    translation_grid_step_m: float,
    translation_grid_range_m: float,
    max_translation_m: float,
    max_rotation_deg: float,
    enable_penetration_term: bool,
    contact_band_m: float,
    huber_delta_m: float,
    w_prior: float,
    max_hand_verts: int,
) -> FrameFit:
    hand_pts = _subsample(
        np.asarray(hand_verts, dtype=np.float64), max_hand_verts, seed=frame_index
    )
    hand_tree = cKDTree(hand_pts)
    hand_hull = None
    if enable_penetration_term and object_mesh is not None and len(hand_pts) >= 8:
        try:
            hand_hull = trimesh.PointCloud(hand_pts).convex_hull
        except Exception:  # degenerate hull (e.g. coplanar hand verts)
            hand_hull = None

    object_pts = np.asarray(object_verts_local, dtype=np.float64)

    def metric_at(pose: np.ndarray) -> tuple[float, float]:
        verts_cam = object_pts @ pose[:3, :3].T + pose[:3, 3]
        return _clearance(hand_pts, verts_cam, hand_hull)

    pre_min, pre_pen = metric_at(source_pose)

    # ---- warmup: translation-only grid search (do-as-i-do warmup_min_clearance)
    steps = np.arange(
        -translation_grid_range_m,
        translation_grid_range_m + 0.5 * translation_grid_step_m,
        translation_grid_step_m,
    )
    gx, gy, gz = np.meshgrid(steps, steps, steps, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    grid = grid[np.linalg.norm(grid, axis=1) <= translation_grid_range_m + 1e-12]
    rotated = object_pts @ source_pose[:3, :3].T
    base_t = source_pose[:3, 3]

    def warmup_cost(offset: np.ndarray) -> float:
        d = hand_tree.query(rotated + (base_t + offset), k=1)[0]
        return float(np.clip(d, 0.0, contact_band_m).mean())

    seed = grid[int(np.argmin([warmup_cost(offset) for offset in grid]))]

    # Correspondence frozen at the warmup seed: object verts whose nearest
    # hand vert lies within the contact band become the contact points the
    # 6-DoF polish refines.
    seed_pose = source_pose.copy()
    seed_pose[:3, 3] = base_t + seed
    seed_cam = object_pts @ seed_pose[:3, :3].T + seed_pose[:3, 3]
    d_seed, nn_seed = hand_tree.query(seed_cam, k=1)
    contact_mask = d_seed <= contact_band_m
    contact_obj_idx = np.flatnonzero(contact_mask)
    contact_hand_idx = nn_seed[contact_mask]

    max_rotation_rad = float(np.radians(max_rotation_deg))
    x0 = np.zeros(6, dtype=np.float64)
    x0[:3] = seed
    result = minimize(
        _contact_objective,
        x0,
        args=(
            object_pts,
            source_pose,
            hand_tree,
            contact_obj_idx,
            contact_hand_idx,
            hand_hull,
            huber_delta_m,
            1.0,
            10.0,
            w_prior,
            max_translation_m,
            max_rotation_rad,
        ),
        method="Powell",
        options={"maxiter": 500, "xtol": 1e-6, "ftol": 1e-9},
    )
    correction = correction_matrix(
        (float(result.x[0]), float(result.x[1]), float(result.x[2])),
        (float(result.x[3]), float(result.x[4]), float(result.x[5])),
    )

    # ---- clamp into the trust region (the optimizer proposes, the gate disposes)
    translation = correction[:3, 3]
    rotvec = Rotation.from_matrix(correction[:3, :3]).as_rotvec()
    t_norm = float(np.linalg.norm(translation))
    r_norm = float(np.linalg.norm(rotvec))
    clamped = t_norm > max_translation_m or np.degrees(r_norm) > max_rotation_deg
    if t_norm > max_translation_m:
        translation = translation * (max_translation_m / t_norm)
    if np.degrees(r_norm) > max_rotation_deg:
        rotvec = rotvec * (max_rotation_rad / r_norm)
    final = correction_matrix(
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (float(rotvec[0]), float(rotvec[1]), float(rotvec[2])),
    )
    post_min, post_pen = metric_at(final @ source_pose)

    return FrameFit(
        frame_index=frame_index,
        pre_min_dist_m=pre_min,
        pre_penetration_depth_m=pre_pen,
        post_min_dist_m=post_min,
        post_penetration_depth_m=post_pen,
        translation_xyz=(
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ),
        rotation_rotvec=(float(rotvec[0]), float(rotvec[1]), float(rotvec[2])),
        clamped=bool(clamped),
        converged=bool(getattr(result, "success", False)),
        objective=float(result.fun) if np.isfinite(result.fun) else float("inf"),
    )


def fit_pose_corrections(
    source_poses: list[np.ndarray],
    frame_indices: list[int],
    hand_verts_list: list[np.ndarray],
    object_verts_local: np.ndarray,
    object_mesh: trimesh.Trimesh | None = None,
    mode: str = "per_frame",
    translation_grid_step_m: float = 0.01,
    translation_grid_range_m: float = 0.06,
    max_translation_m: float = 0.05,
    max_rotation_deg: float = 15.0,
    enable_penetration_term: bool = True,
    contact_band_m: float = 0.03,
    huber_delta_m: float = 0.01,
    w_prior: float = 1.0,
    max_object_verts: int = 2048,
    max_hand_verts: int = 4096,
) -> FitResult:
    """Fit a rigid correction per tracked frame, or one shared correction.

    The three list arguments are parallel and cover only frames with BOTH a
    tracked object pose and at least one valid hand — the overlap evidence.
    ``object_verts_local`` is the metric mesh in the same frame the source
    poses express (FoundationPose's bbox-centred mesh frame).

    ``mode="per_frame"`` fits independently (handles drift).
    ``mode="global"`` fits one shared correction on the concatenated evidence
    — do-as-i-do's trajectory-wide warmup — then validates it per frame so
    the acceptance gates see identical numbers either way.
    """
    if mode not in {"per_frame", "global"}:
        raise ValueError(f"mode must be 'per_frame' or 'global', got {mode!r}")
    if len(source_poses) != len(frame_indices) or len(frame_indices) != len(
        hand_verts_list
    ):
        raise ValueError("source_poses, frame_indices and hand_verts_list must align")
    if not source_poses:
        raise ValueError("no frames to fit")
    if not (0.0 < translation_grid_step_m <= translation_grid_range_m):
        raise ValueError(
            "translation grid step must be positive and no larger than the range"
        )
    if max_translation_m <= 0.0 or max_rotation_deg <= 0.0:
        raise ValueError("max_translation_m and max_rotation_deg must be positive")
    if contact_band_m <= 0.0 or huber_delta_m <= 0.0:
        raise ValueError("contact_band_m and huber_delta_m must be positive")

    object_pts = _subsample(
        np.asarray(object_verts_local, dtype=np.float64), max_object_verts
    )
    if len(object_pts) < 8:
        raise ValueError("object point set too small to fit against")

    common = dict(
        object_verts_local=object_pts,
        object_mesh=object_mesh,
        translation_grid_step_m=translation_grid_step_m,
        translation_grid_range_m=translation_grid_range_m,
        max_translation_m=max_translation_m,
        max_rotation_deg=max_rotation_deg,
        enable_penetration_term=enable_penetration_term,
        contact_band_m=contact_band_m,
        huber_delta_m=huber_delta_m,
        w_prior=w_prior,
        max_hand_verts=max_hand_verts,
    )

    if mode == "per_frame":
        fits = [
            _fit_frame(pose, hand_verts, index, **common)
            for pose, hand_verts, index in zip(
                source_poses, hand_verts_list, frame_indices
            )
        ]
    else:
        all_hands = np.concatenate(
            [
                _subsample(np.asarray(h, dtype=np.float64), max_hand_verts, seed=i)
                for i, h in enumerate(hand_verts_list)
            ]
        )
        shared_fit = _fit_frame(source_poses[0], all_hands, frame_indices[0], **common)
        shared_correction = correction_matrix(
            shared_fit.translation_xyz, shared_fit.rotation_rotvec
        )
        fits = []
        for pose, hand_verts, index in zip(
            source_poses, hand_verts_list, frame_indices
        ):
            hand_pts = _subsample(
                np.asarray(hand_verts, dtype=np.float64), max_hand_verts, seed=index
            )
            hull = None
            if (
                enable_penetration_term
                and object_mesh is not None
                and len(hand_pts) >= 8
            ):
                try:
                    hull = trimesh.PointCloud(hand_pts).convex_hull
                except Exception:
                    hull = None
            pre_min, pre_pen = _clearance(
                hand_pts, object_pts @ pose[:3, :3].T + pose[:3, 3], hull
            )
            post_pose = shared_correction @ pose
            post_min, post_pen = _clearance(
                hand_pts, object_pts @ post_pose[:3, :3].T + post_pose[:3, 3], hull
            )
            fits.append(
                FrameFit(
                    frame_index=index,
                    pre_min_dist_m=pre_min,
                    pre_penetration_depth_m=pre_pen,
                    post_min_dist_m=post_min,
                    post_penetration_depth_m=post_pen,
                    translation_xyz=shared_fit.translation_xyz,
                    rotation_rotvec=shared_fit.rotation_rotvec,
                    clamped=shared_fit.clamped,
                    converged=shared_fit.converged,
                    objective=shared_fit.objective,
                )
            )

    translation_norms = [float(np.linalg.norm(f.translation_xyz)) for f in fits]
    rotation_degs = [float(np.degrees(np.linalg.norm(f.rotation_rotvec))) for f in fits]
    stats = {
        "mode": mode,
        "frame_count": len(fits),
        "pre_min_dist_median_m": float(np.median([f.pre_min_dist_m for f in fits])),
        "post_min_dist_median_m": float(np.median([f.post_min_dist_m for f in fits])),
        "pre_min_dist_max_m": float(np.max([f.pre_min_dist_m for f in fits])),
        "post_min_dist_max_m": float(np.max([f.post_min_dist_m for f in fits])),
        "pre_penetration_max_m": float(
            np.max([f.pre_penetration_depth_m for f in fits])
        ),
        "post_penetration_max_m": float(
            np.max([f.post_penetration_depth_m for f in fits])
        ),
        "translation_norm_mean_m": float(np.mean(translation_norms)),
        "translation_norm_max_m": float(np.max(translation_norms)),
        "rotation_deg_mean": float(np.mean(rotation_degs)),
        "rotation_deg_max": float(np.max(rotation_degs)),
        "clamped_frames": sum(1 for f in fits if f.clamped),
        "unconverged_frames": sum(1 for f in fits if not f.converged),
    }
    return FitResult(mode=mode, frames=tuple(fits), stats=stats)
