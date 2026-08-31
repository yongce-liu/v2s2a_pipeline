"""Trajectory loading and interpolation independent of Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Hand joints and manipulated-object pose in the common world frame."""

    hand_qpos: np.ndarray
    hand_qvel: np.ndarray
    object_pos: np.ndarray
    object_quat_wxyz: np.ndarray
    object_lin_vel: np.ndarray
    object_ang_vel: np.ndarray
    wrist_pos: np.ndarray
    wrist_quat_wxyz: np.ndarray
    finger_qpos: np.ndarray
    finger_qvel: np.ndarray
    fingertip_pos: np.ndarray | None
    contact_schedule: np.ndarray | None
    frequency: float

    @property
    def num_frames(self) -> int:
        return int(self.hand_qpos.shape[0])

    @property
    def hand_dofs(self) -> int:
        return int(self.hand_qpos.shape[1])


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all():
        raise ValueError("quaternion trajectory contains non-finite values")
    if np.any(norms < 1e-8):
        raise ValueError("quaternion trajectory contains a zero quaternion")
    q = q / norms
    q[q[:, 0] < 0] *= -1.0
    return q


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert pipeline quaternions to Isaac Lab's xyzw convention."""
    return np.concatenate((q[..., 1:], q[..., :1]), axis=-1)


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.zeros_like(values)
    velocity[1:] = np.diff(values, axis=0) / dt
    return velocity


def _quat_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler[:, 0] * 0.5, euler[:, 1] * 0.5, euler[:, 2] * 0.5
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return _quat_normalize(
        np.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            axis=-1,
        )
    )


def _quat_angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference wxyz quaternions into world-frame angular velocity."""
    q = _quat_normalize(quat_wxyz.copy())
    # Make the sequence continuous before differencing.
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    # Quaternion multiplication q1 * conjugate(q0), scalar first.
    a, b = q[1:], q[:-1].copy()
    b[:, 1:] *= -1
    w = a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)
    xyz = (
        a[:, :1] * b[:, 1:]
        + b[:, :1] * a[:, 1:]
        + np.cross(a[:, 1:], b[:, 1:])
    )
    angle = 2.0 * np.arctan2(np.linalg.norm(xyz, axis=1), np.clip(w, -1.0, 1.0))
    axis = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12)
    angular = np.zeros((len(q), 3), dtype=q.dtype)
    angular[1:] = axis * angle[:, None] / dt
    return angular


def load_reference(
    path: str | Path,
    hand_dofs: int,
    keypoints_path: str | Path | None = None,
    fingertip_count: int | None = None,
    hand_side: str | None = None,
) -> ReferenceTrajectory:
    """Load a v2s2a ``trajectory_kinematic.npz`` reference."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        if "qpos" not in archive:
            raise ValueError("reference trajectory has no qpos array")
        qpos = np.asarray(archive["qpos"], dtype=np.float32)
        frequency = float(archive["frequency"]) if "frequency" in archive else 50.0
        qvel = np.asarray(archive["qvel"], dtype=np.float32) if "qvel" in archive else None
    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ValueError(f"qpos must have shape (T, D) with T >= 2, got {qpos.shape}")
    if qpos.shape[1] != hand_dofs + 7:
        raise ValueError(f"expected {hand_dofs + 7} qpos columns, got {qpos.shape[1]}")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains non-finite values")
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f"frequency must be finite and positive, got {frequency}")
    if qvel is not None:
        expected_qvel_shape = (qpos.shape[0], hand_dofs + 6)
        if qvel.shape != expected_qvel_shape:
            raise ValueError(f"expected qvel shape {expected_qvel_shape}, got {qvel.shape}")
        if not np.isfinite(qvel).all():
            raise ValueError("qvel contains non-finite values")
    dt = 1.0 / frequency
    hand_qpos = qpos[:, :hand_dofs]
    source_hand_qvel = qvel[:, :hand_dofs] if qvel is not None else _finite_difference(hand_qpos, dt)
    object_pos = qpos[:, hand_dofs : hand_dofs + 3]
    object_quat = _quat_normalize(qpos[:, hand_dofs + 3 : hand_dofs + 7])
    if hand_dofs < 6:
        raise ValueError("the hand trajectory must begin with a 6-DoF wrist")
    wrist_pos = hand_qpos[:, :3]
    wrist_quat = _quat_from_euler_xyz(hand_qpos[:, 3:6])
    # The MJCF wrist hinge rates are not a world-frame angular velocity after
    # conversion to a floating root. Derive the spatial root velocity from pose.
    hand_qvel = np.concatenate(
        (
            _finite_difference(wrist_pos, dt),
            _quat_angular_velocity(wrist_quat, dt),
            source_hand_qvel[:, 6:],
        ),
        axis=-1,
    )
    fingertip_pos = None
    contact_schedule = None
    if keypoints_path:
        with np.load(Path(keypoints_path).expanduser(), allow_pickle=False) as keypoints:
            side = hand_side or ("left" if "qpos_finger_left" in keypoints else "right")
            fingertip_key = f"qpos_finger_{side}"
            if fingertip_key not in keypoints:
                raise ValueError(f"keypoint archive has no {fingertip_key}")
            fingertip_pos = np.asarray(keypoints[fingertip_key], dtype=np.float32)
            if fingertip_pos.ndim != 3 or fingertip_pos.shape[-1] < 3:
                raise ValueError(f"fingertip trajectory must be (T, F, C>=3), got {fingertip_pos.shape}")
            if fingertip_pos.shape[0] < len(hand_qpos):
                raise ValueError(
                    f"fingertip trajectory has {fingertip_pos.shape[0]} frames, "
                    f"expected at least {len(hand_qpos)}"
                )
            # Retargeting's centered valid-window smoothing shortens the IK
            # trajectory. Apply the same centered crop to the source keypoints.
            keypoint_start = (fingertip_pos.shape[0] - len(hand_qpos)) // 2
            expected_fingertips = fingertip_count or fingertip_pos.shape[1]
            if fingertip_pos.shape[1] != expected_fingertips:
                raise ValueError(
                    f"expected {expected_fingertips} fingertips, got {fingertip_pos.shape[1]}"
                )
            fingertip_pos = fingertip_pos[
                keypoint_start : keypoint_start + len(hand_qpos), :, :3
            ]
            explicit_key = f"contact_{side}"
            explicit = (
                np.asarray(keypoints[explicit_key])
                if explicit_key in keypoints
                else np.empty((0, 0))
            )
            # Do not infer contact from fingertip-to-object-origin distance: the
            # origin can be far from the surface and differs across object assets.
            # Missing labels disable contact classification while object-relative
            # fingertip tracking still teaches the demonstrated grasp geometry.
            if explicit.ndim == 2 and explicit.shape[0] >= len(hand_qpos) and explicit.any():
                if explicit.shape[1] % expected_fingertips:
                    raise ValueError(
                        f"contact width {explicit.shape[1]} is not divisible by "
                        f"{expected_fingertips} fingertips"
                    )
                contact_schedule = explicit[
                    keypoint_start : keypoint_start + len(hand_qpos)
                ].reshape(len(hand_qpos), expected_fingertips, -1).any(-1)
    return ReferenceTrajectory(
        hand_qpos=hand_qpos,
        hand_qvel=hand_qvel,
        object_pos=object_pos,
        object_quat_wxyz=object_quat,
        object_lin_vel=_finite_difference(object_pos, dt),
        object_ang_vel=_quat_angular_velocity(object_quat, dt),
        wrist_pos=wrist_pos,
        wrist_quat_wxyz=wrist_quat,
        finger_qpos=hand_qpos[:, 6:],
        finger_qvel=hand_qvel[:, 6:],
        fingertip_pos=fingertip_pos,
        contact_schedule=contact_schedule,
        frequency=frequency,
    )
