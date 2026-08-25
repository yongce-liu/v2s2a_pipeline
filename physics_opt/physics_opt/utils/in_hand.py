"""Hand-object proximity masks (mirrors ``scene_construction.in_hand``): min distance from any MANO hand surface vertex to any
object collision-mesh vertex (world frame at time t).

Hand vertices come from ``mano_verts_{side}`` in the per-task NPZ, shape
``(T, V_hand, 3)``, already world-frame and gravity-aligned by the processor.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation


# Default distance threshold (meters). The real value is configured via
# ``Config.hand_object_distance_thresh`` and
# ``Config.object_pedestal_distance_thresh``; this constant only seeds
# function-signature defaults so ``in_hand.py`` stays usable standalone.
DEFAULT_DISTANCE_THRESH: float = 0.1


def _smooth_endpoint_qpos(qpos: np.ndarray, frame: int, window: int = 3) -> np.ndarray:
    if frame == 0:
        avg = qpos[:window].mean(axis=0).copy()
    elif frame == -1:
        avg = qpos[-window:].mean(axis=0).copy()
    else:
        return qpos[frame]
    quat = avg[3:7]
    n = float(np.linalg.norm(quat))
    if n > 0:
        avg[3:7] = quat / n
    return avg


def _smooth_endpoint_verts(
    verts: np.ndarray, frame: int, window: int = 3
) -> np.ndarray:
    # Matches the IK preprocessor's box filter (solve_ik.py, the `smoothing` branch)
    # so the in-hand check sees the same averaged pose the simulator initializes from.
    if frame == 0:
        return verts[:window].mean(axis=0)
    if frame == -1:
        return verts[-window:].mean(axis=0)
    return verts[frame]


def hand_object_min_dist(
    hand_points: np.ndarray,
    qpos_obj: np.ndarray,
    obj_verts: np.ndarray,
    frame: int = 0,
    smooth_endpoint: bool = False,
) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray]:
    """Min distance from any hand point to any object mesh vertex.

    qpos_obj is ``(T, 7)`` [x, y, z, qw, qx, qy, qz]; obj_verts is mesh-local.
    """
    obj_qpos = (
        _smooth_endpoint_qpos(qpos_obj, frame) if smooth_endpoint else qpos_obj[frame]
    )
    obj_pos = obj_qpos[:3]
    obj_quat_xyzw = obj_qpos[[4, 5, 6, 3]]
    R_obj = Rotation.from_quat(obj_quat_xyzw).as_matrix()
    obj_world = obj_verts @ R_obj.T + obj_pos
    diffs = hand_points[:, None, :] - obj_world[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    flat_idx = int(dists.argmin())
    hand_idx, vert_idx = divmod(flat_idx, dists.shape[1])
    return (
        float(dists[hand_idx, vert_idx]),
        hand_idx,
        hand_points[hand_idx],
        obj_pos,
        obj_world[vert_idx],
    )


def in_hand_at_endpoint(
    hand_verts_world: np.ndarray,
    qpos_obj: np.ndarray,
    obj_verts: np.ndarray,
    frame: int,
    hand_object_distance_thresh: float = DEFAULT_DISTANCE_THRESH,
) -> tuple[bool, float, int, np.ndarray, np.ndarray, np.ndarray]:
    """Endpoint in-hand check with IK-smoothing on both hand and object, so it
    sees the same pose the simulator initializes from. ``frame`` is 0 or -1."""
    hand_pts = _smooth_endpoint_verts(hand_verts_world, frame)
    min_dist, hand_idx, hand_world, obj_pos, vert_world = hand_object_min_dist(
        hand_pts,
        qpos_obj,
        obj_verts,
        frame=frame,
        smooth_endpoint=True,
    )
    in_hand = min_dist < hand_object_distance_thresh
    return in_hand, min_dist, hand_idx, hand_world, obj_pos, vert_world


def find_freeze_indices(mask: np.ndarray) -> tuple[int | None, int | None]:
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return None, None
    return int(idxs[0]), int(idxs[-1])


_MAX_OBJ_VERTS_FOR_MASK: int = 512


def compute_in_hand_mask(
    points_world: np.ndarray,
    qpos_obj: np.ndarray,
    obj_verts: np.ndarray,
    distance_thresh: float = DEFAULT_DISTANCE_THRESH,
    max_obj_verts: int = _MAX_OBJ_VERTS_FOR_MASK,
) -> np.ndarray:
    """Per-frame ``min(dist(points_world[t], obj_world[t])) < thresh`` mask.

    Generic point-set to object-mesh proximity check, used both as an in-hand
    gate (points = hand verts) and an object-near-pedestal gate (points =
    static pedestal centers). Inputs must already share the trajectory timeline.

    ``obj_verts`` is subsampled to ``max_obj_verts`` (seeded): dense
    convex-decomp meshes carry tens of thousands of sub-mm verts but the
    threshold is ~10cm, so coarse coverage suffices. The endpoint check in
    [[in_hand_at_endpoint]] does not subsample (2 frames, exactness matters).
    """
    T = points_world.shape[0]

    if obj_verts.shape[0] > max_obj_verts:
        rng = np.random.default_rng(0)
        idx = rng.choice(obj_verts.shape[0], size=max_obj_verts, replace=False)
        obj_verts = obj_verts[idx]

    obj_pos = qpos_obj[:, :3]  # (T, 3)
    obj_quat_xyzw = qpos_obj[:, [4, 5, 6, 3]]  # (T, 4)
    R_obj = Rotation.from_quat(obj_quat_xyzw).as_matrix()  # (T, 3, 3)

    # Rotate+translate object verts for all frames in one batched op.
    obj_world_all = np.einsum("tij,vj->tvi", R_obj, obj_verts) + obj_pos[:, None, :]
    # (T, V_obj, 3)

    # Per frame: compare squared distances against squared threshold so we skip
    # the per-pair sqrt. cdist with 'sqeuclidean' uses the
    # ‖a-b‖² = ‖a‖² + ‖b‖² - 2 a·bᵀ identity internally, dispatching the inner
    # product to BLAS and avoiding the (M, V_obj, 3) diff tensor that
    # broadcasted-subtract would allocate every frame.
    thresh_sq = float(distance_thresh) ** 2
    out = np.zeros(T, dtype=np.bool_)
    for t in range(T):
        out[t] = (
            cdist(points_world[t], obj_world_all[t], "sqeuclidean").min() < thresh_sq
        )
    return out


def erode_mask(mask: np.ndarray, steps: int) -> np.ndarray:
    """Symmetric temporal erosion of a boolean run mask: ``out[t]`` True iff
    ``mask`` is True across ``[t - steps + 1, t + steps - 1]`` (clamped).

    Shapes the perturbation gate so disturbances fire only well inside a stable
    hold, never during reach-in or set-down transitions.
    """
    if steps <= 1:
        return mask.astype(np.bool_, copy=True)
    T = mask.shape[0]
    out = np.zeros(T, dtype=np.bool_)
    for t in range(T):
        out[t] = bool(mask[max(0, t - steps + 1) : t + steps].all())
    return out


def compute_near_floor_mask(
    qpos_obj: np.ndarray,
    obj_verts: np.ndarray,
    floor_z: float = 0.0,
    distance_thresh: float = DEFAULT_DISTANCE_THRESH,
    max_obj_verts: int = _MAX_OBJ_VERTS_FOR_MASK,
) -> np.ndarray:
    """Per-frame mask: object's lowest world vertex within ``distance_thresh``
    of a horizontal floor plane at ``floor_z``. Floor analog of
    [[compute_in_hand_mask]]'s pedestal gate (a plane has no finite query
    points, so distance is the lowest vertex height above ``floor_z``).
    """
    if obj_verts.shape[0] > max_obj_verts:
        rng = np.random.default_rng(0)
        idx = rng.choice(obj_verts.shape[0], size=max_obj_verts, replace=False)
        obj_verts = obj_verts[idx]

    obj_pos = qpos_obj[:, :3]  # (T, 3)
    obj_quat_xyzw = qpos_obj[:, [4, 5, 6, 3]]  # (T, 4)
    R_obj = Rotation.from_quat(obj_quat_xyzw).as_matrix()  # (T, 3, 3)
    obj_world_all = np.einsum("tij,vj->tvi", R_obj, obj_verts) + obj_pos[:, None, :]
    min_z = obj_world_all[..., 2].min(axis=1)  # (T,)
    return (min_z - floor_z) < float(distance_thresh)


def compute_near_pedestal_mask(
    qpos_obj: np.ndarray,
    obj_verts: np.ndarray,
    ped_pos: np.ndarray,
    ped_radius: np.ndarray,
    ped_half_h: np.ndarray,
    distance_thresh: float = DEFAULT_DISTANCE_THRESH,
    max_obj_verts: int = _MAX_OBJ_VERTS_FOR_MASK,
) -> np.ndarray:
    """Per-frame mask: object's lowest vertex within ``distance_thresh`` above
    a pedestal's top face, restricted to verts over that pedestal's footprint.
    True iff the object is resting on *any* pedestal.

    Pedestal analog of [[compute_near_floor_mask]]. A pedestal is a finite
    cylinder: height is measured above its top face (``center_z + half_h``) and
    only verts whose xy falls within the radius count, else an object passing at
    pedestal-top height elsewhere would register as resting. Collapses to ~0 at
    rest (unlike distance-to-center, which bottoms out at the half-height).
    """
    T = qpos_obj.shape[0]
    if ped_pos.shape[0] == 0:
        return np.zeros(T, dtype=np.bool_)

    if obj_verts.shape[0] > max_obj_verts:
        rng = np.random.default_rng(0)
        idx = rng.choice(obj_verts.shape[0], size=max_obj_verts, replace=False)
        obj_verts = obj_verts[idx]

    obj_pos = qpos_obj[:, :3]  # (T, 3)
    obj_quat_xyzw = qpos_obj[:, [4, 5, 6, 3]]  # (T, 4)
    R_obj = Rotation.from_quat(obj_quat_xyzw).as_matrix()  # (T, 3, 3)
    obj_world_all = np.einsum("tij,vj->tvi", R_obj, obj_verts) + obj_pos[:, None, :]

    thresh = float(distance_thresh)
    out = np.zeros(T, dtype=np.bool_)
    for p in range(ped_pos.shape[0]):
        dxy_sq = ((obj_world_all[..., :2] - ped_pos[p, :2]) ** 2).sum(-1)  # (T, V)
        over = dxy_sq <= ped_radius[p] ** 2  # (T, V)
        height = obj_world_all[..., 2] - (ped_pos[p, 2] + ped_half_h[p])  # (T, V)
        out |= np.where(over, height, np.inf).min(axis=1) < thresh
    return out


def body_has_mesh_geom(model, body_id: int) -> bool:
    """Whether ``body_id`` has a mesh geom: tells a real object body from a
    meshless placeholder (e.g. the bimanual ``left_object`` phantom that
    generate_scene.py emits with only a primitive mass sphere)."""
    import mujoco

    if body_id < 0:
        return False
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_dataid[gi]) >= 0:
            return True
    return False


def extract_body_mesh_verts(
    model,
    body_id: int,
    data=None,
    apply_geom_xform: bool = False,
) -> np.ndarray:
    """Concatenate vertices of every mesh geom attached to ``body_id``.

    Returns mesh-local verts by default. ``apply_geom_xform=True`` applies each
    geom's pos/quat → body-local frame (needed whenever non-identity: every
    robot link, and objects with per-geom-repositioned convex pieces). Passing
    ``data`` (after ``mj_kinematics``) additionally applies the body world
    transform, returning world-frame verts.
    """
    import mujoco
    from physics_opt.utils.mujoco_utils import quat_wxyz_to_rotmat

    parts: list[np.ndarray] = []
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[gi])
        if mesh_id < 0:
            continue
        adr = int(model.mesh_vertadr[mesh_id])
        n = int(model.mesh_vertnum[mesh_id])
        v = np.asarray(model.mesh_vert[adr : adr + n], dtype=np.float64)
        if apply_geom_xform:
            Rg = quat_wxyz_to_rotmat(model.geom_quat[gi])
            v = v @ Rg.T + model.geom_pos[gi]
        parts.append(v)
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert parts, (
        f"Body {body_id!r} ({body_name!r}) has no mesh geoms attached. "
        f"Hand-object distance checks (perturb gate, warmup init) need "
        f"mesh geometry; a primitive-only body would silently degrade them "
        f"to a root-position-only distance. Emit mesh geoms for this body "
        f"in retargeting/pipeline/generate_scene.py."
    )
    out = np.concatenate(parts, axis=0)
    if data is not None:
        out = out @ data.xmat[body_id].reshape(3, 3).T + data.xpos[body_id]
    return out
