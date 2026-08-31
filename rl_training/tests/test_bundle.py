from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v2s2a_rl.bundle import TaskBundle, prepare_task_bundle
from v2s2a_rl.trajectory import load_reference, quat_wxyz_to_xyzw


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    mesh = tmp_path / "run" / "assets" / "objects" / "cube.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    robot = tmp_path / "run" / "assets" / "robots" / "hand.xml"
    robot.parent.mkdir(parents=True)
    robot.write_text("<mujoco/>")
    scene_dir = tmp_path / "run" / "robot" / "hand" / "task" / "0"
    scene_dir.mkdir(parents=True)
    scene = scene_dir / "scene.xml"
    scene.write_text(
        '<mujoco><compiler meshdir="../../../../assets"/><asset>'
        '<mesh name="task_object" file="objects/cube.obj"/></asset><worldbody>'
        '<body name="left_hand_C_MC"><joint name="wrist_x"/><joint name="wrist_y"/>'
        '<joint name="wrist_z"/><joint name="wrist_roll"/>'
        '<joint name="wrist_pitch"/><joint name="wrist_yaw"/>'
        '<joint name="finger_a"/><joint name="finger_b"/>'
        '<body name="digit_alpha_tip_link"><site name="left_alpha_tip"/></body>'
        '<body name="digit_beta_tip_link"><site name="left_beta_tip"/></body>'
        '<body name="left_object"><joint name="object_joint" type="free"/></body>'
        "</body></worldbody></mujoco>"
    )
    qpos = np.zeros((5, 15), dtype=np.float32)
    qpos[:, 6] = np.linspace(0, 1, 5)
    qpos[:, 11] = 1.0  # object wxyz identity after 6 wrist + 2 finger + xyz
    trajectory = scene_dir / "trajectory_kinematic.npz"
    np.savez(trajectory, qpos=qpos, frequency=50.0)
    return trajectory, scene


def test_prepare_roundtrip(tmp_path: Path) -> None:
    trajectory, scene = _fixture(tmp_path)
    output = tmp_path / "bundle.json"
    bundle = prepare_task_bundle(trajectory, scene, output, "pick", scene, trajectory)
    restored = TaskBundle.from_json(output)
    assert restored == bundle
    assert bundle.hand_dofs == 8
    assert bundle.num_frames == 5
    assert bundle.keypoints_path == str(trajectory.resolve())
    assert bundle.object_joint_name == "object_joint"
    assert bundle.object_body_name == "left_object"
    assert bundle.robot_root_body_name == "left_hand_C_MC"
    assert bundle.hand_side == "left"
    assert bundle.finger_joint_names == ("finger_a", "finger_b")
    assert bundle.fingertip_body_names == ("digit_alpha_tip_link", "digit_beta_tip_link")
    assert bundle.fingertip_body_paths == (
        "left_hand_C_MC/digit_alpha_tip_link",
        "left_hand_C_MC/digit_beta_tip_link",
    )
    assert len(bundle.object_asset_paths) == 1


def test_reference_split_and_velocity(tmp_path: Path) -> None:
    trajectory, _ = _fixture(tmp_path)
    reference = load_reference(trajectory, hand_dofs=8)
    assert reference.hand_qpos.shape == (5, 8)
    assert reference.object_pos.shape == (5, 3)
    assert np.allclose(reference.object_quat_wxyz[:, 0], 1.0)
    assert np.allclose(reference.object_lin_vel[1:, 0], 0.0)
    assert reference.finger_qpos.shape == (5, 2)


def test_reference_uses_only_explicit_contacts(tmp_path: Path) -> None:
    trajectory, _ = _fixture(tmp_path)
    keypoints = tmp_path / "keypoints.npz"
    fingertip_pos = np.zeros((5, 5, 3), dtype=np.float32)
    explicit = np.zeros((5, 5), dtype=bool)
    np.savez(keypoints, qpos_finger_left=fingertip_pos, contact_left=explicit)

    reference = load_reference(trajectory, hand_dofs=8, keypoints_path=keypoints)

    assert reference.contact_schedule is None


def test_reference_centers_longer_keypoint_trajectory(tmp_path: Path) -> None:
    trajectory, _ = _fixture(tmp_path)
    keypoints = tmp_path / "keypoints.npz"
    fingertip_pos = np.zeros((7, 2, 3), dtype=np.float32)
    fingertip_pos[:, :, 0] = np.arange(7)[:, None]
    np.savez(keypoints, qpos_finger_left=fingertip_pos)

    reference = load_reference(
        trajectory, hand_dofs=8, keypoints_path=keypoints, fingertip_count=2
    )

    assert reference.fingertip_pos is not None
    assert np.array_equal(reference.fingertip_pos[:, 0, 0], np.arange(1, 6))


def test_reference_preserves_explicit_contacts(tmp_path: Path) -> None:
    trajectory, _ = _fixture(tmp_path)
    keypoints = tmp_path / "keypoints.npz"
    fingertip_pos = np.zeros((5, 3, 3), dtype=np.float32)
    explicit = np.zeros((5, 6), dtype=bool)
    explicit[2, 2:4] = True
    np.savez(keypoints, qpos_finger_right=fingertip_pos, contact_right=explicit)

    reference = load_reference(
        trajectory,
        hand_dofs=8,
        keypoints_path=keypoints,
        fingertip_count=3,
        hand_side="right",
    )

    assert reference.contact_schedule is not None
    assert reference.contact_schedule.shape == (5, 3)
    assert np.array_equal(reference.contact_schedule[2], [False, True, False])


def test_loads_version_one_bundle_with_discovered_fingertips(tmp_path: Path) -> None:
    trajectory, scene = _fixture(tmp_path)
    output = tmp_path / "bundle.json"
    prepare_task_bundle(trajectory, scene, output, scene_usd=scene)
    raw = __import__("json").loads(output.read_text())
    raw["version"] = 1
    raw.pop("finger_joint_names")
    raw.pop("fingertip_body_names")
    raw.pop("fingertip_body_paths")
    raw.pop("hand_side")
    output.write_text(__import__("json").dumps(raw))

    bundle = TaskBundle.from_json(output)

    assert bundle.version == 1
    assert bundle.finger_joint_names == ("finger_a", "finger_b")
    assert bundle.fingertip_body_names == ("digit_alpha_tip_link", "digit_beta_tip_link")
    assert bundle.fingertip_body_paths == (
        "left_hand_C_MC/digit_alpha_tip_link",
        "left_hand_C_MC/digit_beta_tip_link",
    )
    assert bundle.hand_side == "left"


def test_rejects_changed_trajectory(tmp_path: Path) -> None:
    trajectory, scene = _fixture(tmp_path)
    output = tmp_path / "bundle.json"
    prepare_task_bundle(trajectory, scene, output, scene_usd=scene)
    np.savez(trajectory, qpos=np.zeros((5, 15), dtype=np.float32), frequency=50.0)

    with pytest.raises(ValueError, match="trajectory checksum"):
        TaskBundle.from_json(output)


def test_quat_wxyz_to_xyzw() -> None:
    quaternion = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    assert np.array_equal(
        quat_wxyz_to_xyzw(quaternion),
        np.array([[2.0, 3.0, 4.0, 1.0]], dtype=np.float32),
    )


def test_rejects_short_keypoint_trajectory(tmp_path: Path) -> None:
    trajectory, _ = _fixture(tmp_path)
    keypoints = tmp_path / "keypoints.npz"
    np.savez(keypoints, qpos_finger_left=np.zeros((4, 2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="expected at least 5"):
        load_reference(trajectory, hand_dofs=8, keypoints_path=keypoints, fingertip_count=2)


def test_rejects_bad_width(tmp_path: Path) -> None:
    trajectory, scene = _fixture(tmp_path)
    np.savez(trajectory, qpos=np.zeros((4, 8)), frequency=50.0)
    with pytest.raises(ValueError, match="qpos width"):
        prepare_task_bundle(trajectory, scene, tmp_path / "bad.json")


@pytest.mark.parametrize("frequency", [0.0, -1.0, np.inf, np.nan])
def test_reference_rejects_invalid_frequency(tmp_path: Path, frequency: float) -> None:
    trajectory, _ = _fixture(tmp_path)
    with np.load(trajectory) as archive:
        qpos = archive["qpos"]
    np.savez(trajectory, qpos=qpos, frequency=frequency)

    with pytest.raises(ValueError, match="frequency must be finite and positive"):
        load_reference(trajectory, hand_dofs=8)


def test_reference_rejects_bad_qvel_shape(tmp_path: Path) -> None:
    trajectory, _ = _fixture(tmp_path)
    with np.load(trajectory) as archive:
        qpos = archive["qpos"]
    np.savez(trajectory, qpos=qpos, qvel=np.zeros((5, 8)), frequency=50.0)

    with pytest.raises(ValueError, match="expected qvel shape"):
        load_reference(trajectory, hand_dofs=8)


def test_reference_rejects_non_matrix_qpos(tmp_path: Path) -> None:
    trajectory, _ = _fixture(tmp_path)
    np.savez(trajectory, qpos=np.zeros(15), frequency=50.0)

    with pytest.raises(ValueError, match="qpos must have shape"):
        load_reference(trajectory, hand_dofs=8)


def test_reference_rejects_zero_object_quaternion(tmp_path: Path) -> None:
    trajectory, _ = _fixture(tmp_path)
    with np.load(trajectory) as archive:
        qpos = archive["qpos"].copy()
    qpos[:, -4:] = 0.0
    np.savez(trajectory, qpos=qpos, frequency=50.0)

    with pytest.raises(ValueError, match="zero quaternion"):
        load_reference(trajectory, hand_dofs=8)


def test_reference_rejects_non_finite_qpos(tmp_path: Path) -> None:
    trajectory, _ = _fixture(tmp_path)
    with np.load(trajectory) as archive:
        qpos = archive["qpos"].copy()
    qpos[2, 0] = np.nan
    np.savez(trajectory, qpos=qpos, frequency=50.0)

    with pytest.raises(ValueError, match="qpos contains non-finite"):
        load_reference(trajectory, hand_dofs=8)
