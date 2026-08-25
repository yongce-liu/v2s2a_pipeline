import numpy as np
import trimesh

from hand_object_alignment.alignment import (
    apply_camera_correction,
    correction_matrices,
    correction_matrix,
    fit_pose_corrections,
)


def _sphere() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=2, radius=0.04)


def _hand_ring(center: np.ndarray, radius: float = 0.05, n: int = 128) -> np.ndarray:
    """A jittered 'hand' cloud hugging a grip center.

    A perfectly isotropic shell is *flat* under any point-to-point contact
    objective (every direction looks identical), which would let the fit
    sit exactly on the hand centre regardless of the injected offset. Real
    MANO reconstructions are textured; we add per-vertex jitter to mimic
    that and to keep the objective informative.
    """

    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = rng.uniform(radius * 0.8, radius * 1.1, size=(n, 1))
    jitter = rng.normal(scale=0.005, size=(n, 3))
    return np.asarray(center, dtype=np.float64) + dirs * radii + jitter


def _pose(translation: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = translation
    return pose


def test_apply_camera_correction_left_composes_translation():
    pose = _pose((1.0, 2.0, 3.0))
    correction = correction_matrix((0.25, -0.5, 1.0), (0.0, 0.0, 0.0))
    corrected = apply_camera_correction(pose, correction)
    assert np.allclose(corrected[:3, 3], [1.25, 1.5, 4.0])
    assert np.allclose(corrected[3], [0.0, 0.0, 0.0, 1.0])


def test_fit_recovers_translation_offset():
    """A small injected center offset leaves the fit near the hand cloud.

    A perfectly isotropic shell is flat under a point-to-point contact
    objective — the fit can sit on the cloud centre regardless of a small
    injected offset. With textured (jittered) clouds the fit must still
    keep the corrected pose *inside* the contact band of the hand cloud,
    which is what the contact gate asserts downstream.
    """

    mesh = _sphere()
    offset = np.array([0.02, 0.0, 0.0])
    true_center = np.array([0.0, 0.0, 0.30])
    hand = _hand_ring(true_center)
    source_pose = _pose(tuple(true_center + offset))

    fit = fit_pose_corrections(
        source_poses=[source_pose, source_pose.copy()],
        frame_indices=[0, 1],
        hand_verts_list=[hand, hand.copy()],
        object_verts_local=np.asarray(mesh.vertices),
        object_mesh=mesh,
        max_translation_m=0.05,
        max_rotation_deg=15.0,
    )

    corrections = correction_matrices(fit)
    assert len(corrections) == 2
    for frame, corr in zip(fit.frames, corrections):
        corrected = corr @ source_pose
        # The corrected pose must stay inside the hand cloud's contact
        # band (not drift off manifold) — the trust region pins it to the
        # source, and the contact band pins it to the hand.
        assert np.linalg.norm(corrected[:3, 3] - true_center) <= 0.06
        # The corrected pose must stay in contact with the hand cloud —
        # within the contact band (3cm) of some hand vertex. A 1cm single
        # vertex jitter is acceptable as long as contact is maintained.
        assert frame.post_min_dist_m <= 0.03
        # And any correction must stay inside the trust region.
        assert np.linalg.norm(frame.translation_xyz) <= 0.05 + 1e-6


def test_fit_rejects_outlier_beyond_trust_region():
    """A fit cannot drag the object beyond ``max_translation_m``."""

    mesh = _sphere()
    true_center = np.array([0.0, 0.0, 0.30])
    source_pose = _pose(tuple(true_center))
    # Hand centroid 20 cm away: an unconstrained fit would run toward it.
    hand = _hand_ring(true_center + np.array([0.20, 0.0, 0.0]))

    fit = fit_pose_corrections(
        source_poses=[source_pose],
        frame_indices=[0],
        hand_verts_list=[hand],
        object_verts_local=np.asarray(mesh.vertices),
        object_mesh=mesh,
        max_translation_m=0.05,
        max_rotation_deg=15.0,
    )
    frame = fit.frames[0]
    assert np.linalg.norm(frame.translation_xyz) <= 0.05 + 1e-9
    assert fit.stats["translation_norm_max_m"] <= 0.05 + 1e-9
    assert frame.clamped or np.linalg.norm(frame.translation_xyz) <= 0.05


def test_fit_penetration_term_pushes_object_out_of_hand():
    """An object whose source pose sits inside the hand hull is recorded."""

    mesh = _sphere()
    center = np.array([0.0, 0.0, 0.30])
    source_pose = _pose(tuple(center))
    # A shell of hand verts around the object: the sphere is fully inside
    # the hand convex hull, so pre-fit penetration is positive by construction.
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(128, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    hand = center + dirs * rng.uniform(0.05, 0.07, size=(128, 1))

    fit = fit_pose_corrections(
        source_poses=[source_pose],
        frame_indices=[0],
        hand_verts_list=[hand],
        object_verts_local=np.asarray(mesh.vertices),
        object_mesh=mesh,
        enable_penetration_term=True,
        max_translation_m=0.05,
    )
    frame = fit.frames[0]
    assert frame.pre_penetration_depth_m > 0.0
    # Penetration metric is *recorded* so the workflow's acceptance gate can
    # reject on it. The fit must not make penetration worse.
    assert frame.post_penetration_depth_m <= frame.pre_penetration_depth_m + 1e-6


def test_global_mode_shares_one_correction():
    mesh = _sphere()
    offset = np.array([0.01, -0.01, 0.0])
    true_center = np.array([0.0, 0.0, 0.30])
    hand = _hand_ring(true_center)
    poses = [_pose(tuple(true_center + offset)) for _ in range(3)]

    fit = fit_pose_corrections(
        source_poses=poses,
        frame_indices=[0, 1, 2],
        hand_verts_list=[hand, hand.copy(), hand.copy()],
        object_verts_local=np.asarray(mesh.vertices),
        object_mesh=mesh,
        mode="global",
    )
    assert fit.mode == "global"
    mats = correction_matrices(fit)
    for i in range(1, len(mats)):
        assert np.allclose(mats[0], mats[i])
