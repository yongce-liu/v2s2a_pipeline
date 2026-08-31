"""Direct Isaac Lab environment for learning robust residual trajectory tracking."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.schemas import activate_contact_sensors
from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from v2s2a_rl.bundle import TaskBundle
from v2s2a_rl.trajectory import load_reference, quat_wxyz_to_xyzw

from .trajectory_env_cfg import TrajectoryTrackingEnvCfg


class TrajectoryTrackingEnv(DirectRLEnv):
    """Track hand and object demonstrations with residual control.

    Design follows the strongest reusable ideas in robotic_grounding and
    IsaacLab dexterous tasks: demonstration-conditioned observations, residual
    actions, random phase resets, teacher/action mixing, object assistance that
    decays to zero, domain randomization and explicit task-success metrics.
    """

    cfg: TrajectoryTrackingEnvCfg

    def __init__(self, cfg: TrajectoryTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        bundle = TaskBundle.from_json(cfg.bundle_path)
        reference = load_reference(
            bundle.trajectory_path,
            bundle.hand_dofs,
            bundle.keypoints_path or None,
            len(bundle.fingertip_body_names) or None,
            bundle.hand_side,
        )

        self._ref_q = torch.as_tensor(reference.hand_qpos, device=self.device)
        self._ref_qd = torch.as_tensor(reference.hand_qvel, device=self.device)
        self._ref_wrist_pos = torch.as_tensor(reference.wrist_pos, device=self.device)
        self._ref_wrist_quat = torch.as_tensor(
            quat_wxyz_to_xyzw(reference.wrist_quat_wxyz), device=self.device
        )
        self._ref_finger_q = torch.as_tensor(reference.finger_qpos, device=self.device)
        self._ref_finger_qd = torch.as_tensor(reference.finger_qvel, device=self.device)
        self._ref_fingertips = (
            torch.as_tensor(reference.fingertip_pos, device=self.device)
            if reference.fingertip_pos is not None
            else None
        )
        self._ref_contact = (
            torch.as_tensor(reference.contact_schedule, device=self.device, dtype=torch.bool)
            if reference.contact_schedule is not None
            else None
        )
        self._ref_obj_pos = torch.as_tensor(reference.object_pos, device=self.device)
        self._ref_obj_quat = torch.as_tensor(
            quat_wxyz_to_xyzw(reference.object_quat_wxyz), device=self.device
        )
        self._ref_fingertip_object_offset = (
            quat_apply(
                quat_inv(self._ref_obj_quat).unsqueeze(1).expand(-1, self._ref_fingertips.shape[1], -1),
                self._ref_fingertips - self._ref_obj_pos.unsqueeze(1),
            )
            if self._ref_fingertips is not None
            else None
        )
        self._ref_obj_linvel = torch.as_tensor(reference.object_lin_vel, device=self.device)
        self._ref_obj_angvel = torch.as_tensor(reference.object_ang_vel, device=self.device)
        self._ref_obj_relative_to_wrist_pos = quat_apply(
            quat_inv(self._ref_wrist_quat), self._ref_obj_pos - self._ref_wrist_pos
        )
        self._ref_obj_relative_to_wrist_quat = quat_mul(
            quat_inv(self._ref_wrist_quat), self._ref_obj_quat
        )
        self._horizon = reference.num_frames
        self._hand_dofs = bundle.hand_dofs

        finger_names = bundle.finger_joint_names
        self._joint_name_to_id = {name: i for i, name in enumerate(self.robot.joint_names)}
        missing = [name for name in finger_names if name not in self._joint_name_to_id]
        if missing:
            raise RuntimeError(f"converted USD did not preserve finger joints: {missing}")
        self._finger_ids = torch.tensor(
            [self._joint_name_to_id[name] for name in finger_names],
            device=self.device,
            dtype=torch.int32,
        )

        body_name_to_id = {name: index for index, name in enumerate(self.robot.body_names)}
        missing_fingertips = [
            name for name in bundle.fingertip_body_names if name not in body_name_to_id
        ]
        if missing_fingertips:
            raise RuntimeError(
                f"converted USD did not preserve fingertip bodies: {missing_fingertips}"
            )
        if self._ref_fingertips is not None and not bundle.fingertip_body_names:
            raise RuntimeError("keypoint trajectory requires fingertip_body_names in the task bundle")
        if self._ref_fingertips is not None and self._ref_fingertips.shape[1] != len(
            bundle.fingertip_body_names
        ):
            raise RuntimeError(
                "fingertip keypoint count does not match task bundle: "
                f"{self._ref_fingertips.shape[1]} != {len(bundle.fingertip_body_names)}"
            )
        self._fingertip_ids = torch.tensor(
            [body_name_to_id[name] for name in bundle.fingertip_body_names],
            device=self.device,
            dtype=torch.int32,
        )

        self._actions = torch.zeros(self.num_envs, self._hand_dofs, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._action_delta = torch.zeros_like(self._actions)
        self._targets = torch.zeros_like(self._actions)
        self._action_scale = torch.full(
            (self._hand_dofs,), self.cfg.finger_action_scale, device=self.device
        )
        self._action_scale[:3] = self.cfg.wrist_position_action_scale
        self._action_scale[3:6] = self.cfg.wrist_rotation_action_scale
        self._root_body_ids = torch.zeros(1, device=self.device, dtype=torch.long)
        self._wrist_target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._frame = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._reference_time = torch.zeros(self.num_envs, device=self.device)
        self._reference_frame_dt = 1.0 / reference.frequency
        self._global_steps = 0
        self._reset_difficulty = float(cfg.curriculum_initial_difficulty)
        self._curriculum_success_ema = 0.0
        self._episode_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._success_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._object_assist = float(cfg.object_assist_initial)
        self._nominal_stiffness = self.robot.data.joint_stiffness.torch[:, self._finger_ids].clone()
        self._nominal_damping = self.robot.data.joint_damping.torch[:, self._finger_ids].clone()
        # The base constructor cannot call the task reset before reference data is loaded.
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self) -> None:
        if self.cfg.scene_asset is None:
            raise RuntimeError("scene_asset was not initialized from the task bundle")
        self.robot = Articulation(self.cfg.scene_asset)
        if self.cfg.object_asset is None:
            raise RuntimeError("object_asset was not initialized from the task bundle")
        self.object = RigidObject(self.cfg.object_asset)
        if self.cfg.fingertip_contacts is None:
            raise RuntimeError("fingertip_contacts were not initialized from the task bundle")
        for sensor_cfg in self.cfg.fingertip_contacts.values():
            source_path = sensor_cfg.prim_path.replace("env_.*", "env_0")
            activate_contact_sensors(source_path, threshold=self.cfg.contact_force_threshold)
        self.fingertip_contacts = {
            name: ContactSensor(sensor_cfg)
            for name, sensor_cfg in self.cfg.fingertip_contacts.items()
        }
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["scene"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        for name, sensor in self.fingertip_contacts.items():
            self.scene.sensors[f"fingertip_contact_{name}"] = sensor

        light = __import__("isaaclab.sim", fromlist=["DomeLightCfg"]).DomeLightCfg(
            # Keep the white hand geometry from washing out against the default
            # light background in recorded headless videos.
            intensity=800.0, color=(0.75, 0.78, 0.85)
        )
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._global_steps += 1
        # RslRlVecEnvWrapper clips to the same range before PPO stores the
        # transition, so the action optimized by PPO is exactly the one executed.
        self._actions[:] = actions
        residual = self.cfg.action_ema * self._previous_actions + (1.0 - self.cfg.action_ema) * self._actions
        self._action_delta[:] = residual - self._previous_actions
        self._previous_actions[:] = residual
        reference = self._ref_q[self._frame]
        self._targets[:] = reference + self._action_scale * residual
        rotation_residual = self.cfg.wrist_rotation_action_scale * residual[:, 3:6]
        residual_quat = quat_from_euler_xyz(
            rotation_residual[:, 0], rotation_residual[:, 1], rotation_residual[:, 2]
        )
        self._wrist_target_quat[:] = quat_mul(
            residual_quat, self._ref_wrist_quat[self._frame]
        )

        # Assistance follows demonstrated competence instead of wall-clock
        # steps. Decaying it before the tracker succeeds created a moving target
        # and stalled earlier runs at near-zero success.
        self._object_assist = self.cfg.object_assist_initial * (
            1.0 - self._reset_difficulty
        )

    def _apply_action(self) -> None:
        wrist_target_pos = self._targets[:, :3]
        wrist_target_quat = self._wrist_target_quat
        wrist_pos = self.robot.data.root_pos_w.torch - self.scene.env_origins
        wrist_quat = self.robot.data.root_quat_w.torch
        wrist_velocity = self.robot.data.root_vel_w.torch
        wrist_force = (
            self.cfg.wrist_position_stiffness * (wrist_target_pos - wrist_pos)
            - self.cfg.wrist_velocity_damping * wrist_velocity[:, :3]
        )
        wrist_torque = (
            self.cfg.wrist_rotation_stiffness
            * axis_angle_from_quat(quat_mul(wrist_target_quat, quat_inv(wrist_quat)))
            - self.cfg.wrist_angular_damping * wrist_velocity[:, 3:]
        )
        wrist_force_body = torch.clamp(
            quat_apply(quat_inv(wrist_quat), wrist_force),
            -self.cfg.wrist_max_force,
            self.cfg.wrist_max_force,
        )
        wrist_torque_body = torch.clamp(
            quat_apply(quat_inv(wrist_quat), wrist_torque),
            -self.cfg.wrist_max_torque,
            self.cfg.wrist_max_torque,
        )
        self.robot.permanent_wrench_composer.reset()
        self.robot.permanent_wrench_composer.add_forces_and_torques_index(
            wrist_force_body.unsqueeze(1),
            wrist_torque_body.unsqueeze(1),
            body_ids=self._root_body_ids,
            is_global=False,
        )
        self.robot.set_joint_position_target_index(
            target=self._targets[:, 6:],
            joint_ids=self._finger_ids,
        )
        if self._object_assist <= 0:
            self.object.permanent_wrench_composer.reset()
            return
        current_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        current_quat = self.object.data.root_quat_w.torch
        current_velocity = self.object.data.root_vel_w.torch
        target_pos = self._ref_obj_pos[self._frame]
        target_quat = self._ref_obj_quat[self._frame]
        position_error = target_pos - current_pos
        orientation_error = axis_angle_from_quat(quat_mul(target_quat, quat_inv(current_quat)))
        force_world = (
            self.cfg.object_assist_position_stiffness * position_error
            + self.cfg.object_assist_velocity_damping
            * (self._ref_obj_linvel[self._frame] - current_velocity[:, :3])
        )
        torque_world = (
            self.cfg.object_assist_rotation_stiffness * orientation_error
            + self.cfg.object_assist_angular_damping
            * (self._ref_obj_angvel[self._frame] - current_velocity[:, 3:])
        )
        force_body = quat_apply(quat_inv(current_quat), force_world)
        torque_body = quat_apply(quat_inv(current_quat), torque_world)
        force_body = torch.clamp(
            self._object_assist * force_body,
            -self.cfg.object_assist_force,
            self.cfg.object_assist_force,
        )
        torque_body = torch.clamp(
            self._object_assist * torque_body,
            -self.cfg.object_assist_torque,
            self.cfg.object_assist_torque,
        )
        self.object.permanent_wrench_composer.reset()
        self.object.permanent_wrench_composer.add_forces_and_torques_index(
            force_body.unsqueeze(1), torque_body.unsqueeze(1), is_global=False
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._compute_state()
        phase = self._frame.float().unsqueeze(-1) / max(1, self._horizon - 1)
        phase_encoding = torch.cat((torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)), dim=-1)
        q_ref = self._ref_q[self._frame]
        qd_ref = self._ref_qd[self._frame]
        obj_ref_pos = self._ref_obj_pos[self._frame]
        obj_ref_quat = self._ref_obj_quat[self._frame]
        wrist_pos = self._hand_q[:, :3]
        wrist_quat = self.robot.data.root_quat_w.torch
        object_relative_to_wrist_pos = quat_apply(
            quat_inv(wrist_quat), self._object_pos - wrist_pos
        )
        object_relative_to_wrist_quat = quat_mul(
            quat_inv(wrist_quat), self._object_quat
        )
        chunks = [
            phase_encoding,
            self._hand_q,
            self._hand_qd,
            q_ref - self._hand_q,
            qd_ref - self._hand_qd,
            self._object_pos,
            self._object_quat,
            self._object_linvel,
            self._object_angvel,
            obj_ref_pos - self._object_pos,
            quat_mul(obj_ref_quat, quat_inv(self._object_quat)),
            object_relative_to_wrist_pos,
            object_relative_to_wrist_quat,
            self._ref_obj_relative_to_wrist_pos[self._frame]
            - object_relative_to_wrist_pos,
            quat_mul(
                self._ref_obj_relative_to_wrist_quat[self._frame],
                quat_inv(object_relative_to_wrist_quat),
            ),
            self._previous_actions,
        ]
        for offset in self.cfg.future_command_steps:
            future = torch.clamp(self._frame + offset, max=self._horizon - 1)
            chunks.extend(
                (
                    self._ref_q[future] - self._hand_q,
                    self._ref_obj_pos[future] - self._object_pos,
                    quat_mul(self._ref_obj_quat[future], quat_inv(self._object_quat)),
                )
            )
        observation = torch.cat(chunks, dim=-1)
        observation = torch.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
        observation = torch.clamp(observation, -10.0, 10.0)
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_state()
        hand_q = torch.nan_to_num(self._hand_q, nan=0.0, posinf=10.0, neginf=-10.0)
        hand_q = torch.clamp(hand_q, -10.0, 10.0)
        hand_qd = torch.nan_to_num(self._hand_qd, nan=0.0, posinf=100.0, neginf=-100.0)
        hand_qd = torch.clamp(hand_qd, -100.0, 100.0)
        object_pos = torch.nan_to_num(self._object_pos, nan=1e3, posinf=1e3, neginf=-1e3)
        object_quat = torch.nan_to_num(self._object_quat, nan=0.0)
        invalid_quat = torch.linalg.norm(object_quat, dim=-1, keepdim=True) < 1e-6
        identity = torch.zeros_like(object_quat)
        identity[:, 3] = 1.0
        object_quat = torch.where(invalid_quat, identity, object_quat)
        object_quat = object_quat / torch.linalg.norm(object_quat, dim=-1, keepdim=True).clamp(min=1e-6)
        q_error = torch.mean((hand_q - self._ref_q[self._frame]) ** 2, dim=-1)
        qd_error = torch.mean((hand_qd - self._ref_qd[self._frame]) ** 2, dim=-1)
        pos_error = torch.linalg.norm(object_pos - self._ref_obj_pos[self._frame], dim=-1)
        rot_error = quat_error_magnitude(object_quat, self._ref_obj_quat[self._frame])
        wrist_pos = torch.clamp(hand_q[:, :3], -10.0, 10.0)
        wrist_quat = torch.nan_to_num(self.robot.data.root_quat_w.torch, nan=0.0)
        invalid_wrist_quat = torch.linalg.norm(wrist_quat, dim=-1, keepdim=True) < 1e-6
        wrist_quat = torch.where(invalid_wrist_quat, identity, wrist_quat)
        wrist_quat = wrist_quat / torch.linalg.norm(
            wrist_quat, dim=-1, keepdim=True
        ).clamp(min=1e-6)
        object_relative_to_wrist_pos = quat_apply(
            quat_inv(wrist_quat), torch.clamp(object_pos - wrist_pos, -10.0, 10.0)
        )
        object_relative_to_wrist_quat = quat_mul(quat_inv(wrist_quat), object_quat)
        relative_pos_error = torch.linalg.norm(
            object_relative_to_wrist_pos - self._ref_obj_relative_to_wrist_pos[self._frame],
            dim=-1,
        )
        relative_pos_error = torch.nan_to_num(
            relative_pos_error, nan=1e3, posinf=1e3, neginf=1e3
        )
        relative_rot_error = quat_error_magnitude(
            object_relative_to_wrist_quat,
            self._ref_obj_relative_to_wrist_quat[self._frame],
        )
        action_rate = torch.mean(self._action_delta**2, dim=-1)

        hand_reward = torch.exp(-q_error / self.cfg.hand_position_sigma)
        object_pos_reward = torch.exp(-(pos_error**2) / self.cfg.object_position_sigma**2)
        object_rot_reward = torch.exp(-(rot_error**2) / self.cfg.object_rotation_sigma**2)
        relative_pos_reward = torch.exp(
            -(relative_pos_error**2) / self.cfg.relative_position_sigma**2
        )
        relative_rot_reward = torch.exp(
            -(relative_rot_error**2) / self.cfg.relative_rotation_sigma**2
        )
        # The terminal objective is only relevant near the demonstrated end.
        # Rewarding it from frame zero encourages shortcuts that skip the grasp.
        terminal_phase = torch.clamp(
            (self._frame.float() / max(1, self._horizon - 1) - self.cfg.goal_reward_start_phase)
            / max(1e-6, 1.0 - self.cfg.goal_reward_start_phase),
            0.0,
            1.0,
        )
        goal_pos_error = torch.linalg.norm(
            object_pos - self._ref_obj_pos[-1].unsqueeze(0), dim=-1
        )
        goal_progress = terminal_phase * torch.clamp(
            1.0 - goal_pos_error / max(self.cfg.early_termination_position_error, 1e-3),
            0.0,
            1.0,
        )
        relative_rot_error = torch.nan_to_num(
            relative_rot_error, nan=torch.pi, posinf=torch.pi, neginf=torch.pi
        )
        fingertip_reward, contact_match, missed_contact, unintended_contact = (
            self._contact_and_keypoint_rewards()
        )
        fingertip_reward = torch.nan_to_num(fingertip_reward, nan=0.0)
        # Keep a dense gradient while contact/fingertip tracking is initially
        # poor. The previous geometric mean collapsed to roughly 1e-3 whenever
        # one term approached zero and left PPO with almost no useful signal.
        tracking_terms = torch.stack(
            (
                hand_reward,
                fingertip_reward,
                object_pos_reward,
                object_rot_reward,
                relative_pos_reward,
                relative_rot_reward,
            ),
            dim=-1,
        )
        tracking_terms = torch.nan_to_num(tracking_terms, nan=0.0, posinf=1.0, neginf=0.0)
        tracking_reward = torch.sum(
            tracking_terms
            * torch.tensor(
                [1.0, 2.0, 4.0, 1.5, 2.0, 1.0],
                device=self.device,
            ),
            dim=-1,
        ) / 11.5
        reward = (
            10.0 * tracking_reward
            + 2.0 * object_pos_reward
            + 1.0 * object_rot_reward
            + 2.0 * goal_progress
            + 4.0 * contact_match
            - 3.0 * missed_contact
            - 1.0 * unintended_contact
            - 0.002 * torch.mean(self._actions**2, dim=-1)
            - 0.005 * action_rate
            - 0.001 * torch.clamp(qd_error, max=100.0)
        )

        at_end = self._frame >= self._horizon - 2
        success = at_end & (pos_error < self.cfg.success_position_tolerance) & (
            rot_error < self.cfg.success_rotation_tolerance
        )
        self._episode_success |= success
        reward += success.float() * 25.0
        self.extras.setdefault("log", {})
        self.extras["log"].update(
            {
                "Metrics/object_position_error": pos_error.mean(),
                "Metrics/object_rotation_error": rot_error.mean(),
                "Metrics/hand_joint_rmse": torch.sqrt(q_error).mean(),
                "Metrics/relative_position_error": relative_pos_error.mean(),
                "Metrics/relative_rotation_error": relative_rot_error.mean(),
                "Metrics/tracking_geometric_mean": tracking_reward.mean(),
                "Metrics/fingertip_tracking": fingertip_reward.mean(),
                "Metrics/contact_match": contact_match.mean(),
                "Metrics/missed_contact": missed_contact.mean(),
                "Metrics/unintended_contact": unintended_contact.mean(),
                "Metrics/success_rate_step": success.float().mean(),
                "Curriculum/object_assist": self._object_assist,
                "Curriculum/reset_difficulty": self._reset_difficulty,
                "Curriculum/success_ema": self._curriculum_success_ema,
            }
        )
        self._reference_time += self.cfg.control_dt * self.cfg.reference_speed
        self._frame = torch.clamp(
            torch.floor(self._reference_time / self._reference_frame_dt).long(),
            max=self._horizon - 1,
        )
        reward = torch.nan_to_num(reward, nan=-10.0, posinf=25.0, neginf=-10.0)
        return torch.clamp(reward, -10.0, 32.0)

    def _contact_and_keypoint_rewards(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Port robotic_grounding's keypoint/contact schedule rewards.

        Fingertip tracking remains active without contact labels. When labels
        exist, one filtered PhysX sensor per fingertip measures object contact.
        """
        zeros = torch.zeros(self.num_envs, device=self.device)
        if self._ref_fingertips is None:
            return zeros, zeros, zeros, zeros
        current_tip = self.robot.data.body_pos_w.torch[:, self._fingertip_ids]
        current_tip = current_tip - self.scene.env_origins.unsqueeze(1)
        # Track the demonstrated grasp in the live object frame. This remains
        # meaningful when the object drifts from its world-space reference.
        object_quat = self._object_quat.unsqueeze(1).expand(
            -1, self._ref_fingertip_object_offset.shape[1], -1
        )
        target_tip = self._object_pos.unsqueeze(1) + quat_apply(
            object_quat, self._ref_fingertip_object_offset[self._frame]
        )
        tip_error_sq = (current_tip - target_tip).square().sum(dim=-1)
        fingertip_reward = torch.exp(
            -tip_error_sq / (self.cfg.hand_keypoint_sigma**2)
        ).mean(dim=-1)

        if self._ref_contact is None:
            return fingertip_reward, zeros, zeros, zeros
        fingertip_forces = []
        for sensor in self.fingertip_contacts.values():
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                fingertip_forces.append(
                    torch.zeros(self.num_envs, 3, device=self.device)
                )
            else:
                fingertip_forces.append(force_matrix.torch[:, 0].sum(dim=1))
        force = torch.stack(fingertip_forces, dim=1)
        current_contact = (
            torch.linalg.norm(force, dim=-1) > self.cfg.contact_force_threshold
        )

        expected = self._ref_contact[self._frame]
        expected_count = expected.sum(dim=-1).float().clamp(min=1.0)
        contact_match = (current_contact & expected).sum(dim=-1).float() / expected_count
        missed = (expected & ~current_contact).sum(dim=-1).float() / expected_count
        unexpected_count = (~expected).sum(dim=-1).float().clamp(min=1.0)
        unintended = (current_contact & ~expected).sum(dim=-1).float() / unexpected_count
        active = expected.any(dim=-1).float()
        return fingertip_reward, contact_match * active, missed * active, unintended

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_state()
        object_pos = torch.nan_to_num(self._object_pos, nan=1e3, posinf=1e3, neginf=-1e3)
        pos_error = torch.linalg.norm(object_pos - self._ref_obj_pos[self._frame], dim=-1)
        failed = pos_error > self.cfg.early_termination_position_error
        timed_out = self._frame >= self._horizon - 1
        return failed, timed_out

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        # DirectRLEnv can request a reset during base construction, before this
        # subclass has loaded its task bundle. Defer that first reset.
        if not hasattr(self, "_ref_q"):
            return
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.int32)
        terminal_success = self._episode_success[env_ids].clone()
        self.extras["terminal_success"] = terminal_success
        episode_success_rate = terminal_success.float().mean()
        self.extras.setdefault("log", {})["Metrics/episode_success_rate"] = episode_success_rate
        if self._global_steps > 0 and len(env_ids) >= self.cfg.curriculum_min_completed_episodes:
            batch_success = float(episode_success_rate.item())
            smoothing = self.cfg.curriculum_success_smoothing
            self._curriculum_success_ema = (
                smoothing * self._curriculum_success_ema + (1.0 - smoothing) * batch_success
            )
            if self._curriculum_success_ema >= self.cfg.curriculum_success_threshold:
                self._reset_difficulty = min(
                    1.0, self._reset_difficulty + self.cfg.curriculum_difficulty_step
                )
            elif self._curriculum_success_ema <= self.cfg.curriculum_failure_threshold:
                self._reset_difficulty = max(
                    self.cfg.curriculum_initial_difficulty,
                    self._reset_difficulty - self.cfg.curriculum_difficulty_step,
                )
        self._episode_success[env_ids] = False
        reset_frame_fraction = (
            self.cfg.reset_frame_fraction
            + (self.cfg.curriculum_final_reset_frame_fraction - self.cfg.reset_frame_fraction)
            * self._reset_difficulty
        )
        max_start = int((self._horizon - 2) * reset_frame_fraction)
        min_start = min(
            max_start,
            int((self._horizon - 2) * self.cfg.curriculum_min_reset_phase),
        )
        self._frame[env_ids] = (
            torch.randint(
                min_start,
                max(min_start + 1, max_start + 1),
                (len(env_ids),),
                device=self.device,
            )
            if max_start > 0
            else 0
        )
        first_frame_probability = (
            self.cfg.reset_first_frame_probability
            + (
                self.cfg.curriculum_final_first_frame_probability
                - self.cfg.reset_first_frame_probability
            )
            * self._reset_difficulty
        )
        first_frame = torch.rand(len(env_ids), device=self.device) < first_frame_probability
        self._frame[env_ids] = torch.where(
            first_frame, torch.zeros_like(self._frame[env_ids]), self._frame[env_ids]
        )
        self._reference_time[env_ids] = self._frame[env_ids].float() * self._reference_frame_dt

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        joint_pos[:, self._finger_ids] = self._ref_finger_q[self._frame[env_ids]]
        joint_vel[:, self._finger_ids] = self._ref_finger_qd[self._frame[env_ids]]
        noise_scale = self.cfg.curriculum_min_reset_noise_scale + (
            1.0 - self.cfg.curriculum_min_reset_noise_scale
        ) * self._reset_difficulty
        noise = noise_scale * self.cfg.joint_reset_noise * (
            2.0 * torch.rand_like(joint_pos[:, self._finger_ids]) - 1.0
        )
        joint_pos[:, self._finger_ids] += noise
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target_index(
            target=joint_pos[:, self._finger_ids], joint_ids=self._finger_ids, env_ids=env_ids
        )
        wrist_pose = torch.cat(
            (
                self._ref_wrist_pos[self._frame[env_ids]] + self.scene.env_origins[env_ids],
                self._ref_wrist_quat[self._frame[env_ids]],
            ),
            dim=-1,
        )
        wrist_vel = torch.zeros(len(env_ids), 6, device=self.device)
        self.robot.write_root_pose_to_sim_index(root_pose=wrist_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=wrist_vel, env_ids=env_ids)
        stiffness_scale = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.joint_stiffness_scale_range
        )
        damping_scale = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.cfg.joint_damping_scale_range
        )
        self.robot.write_joint_stiffness_to_sim_index(
            stiffness=self._nominal_stiffness[env_ids] * stiffness_scale,
            joint_ids=self._finger_ids,
            env_ids=env_ids,
        )
        self.robot.write_joint_damping_to_sim_index(
            damping=self._nominal_damping[env_ids] * damping_scale,
            joint_ids=self._finger_ids,
            env_ids=env_ids,
        )

        # Set the floating object's body pose through articulation root/body state
        # on reset when supported by the selected backend. The object assist term
        # then closes residual importer/backend discrepancies during curriculum.
        object_position = self._ref_obj_pos[self._frame[env_ids]] + self.scene.env_origins[env_ids]
        object_position += noise_scale * self.cfg.object_reset_position_noise * (
            2.0 * torch.rand_like(object_position) - 1.0
        )
        rotation_noise = noise_scale * self.cfg.object_reset_rotation_noise * (
            2.0 * torch.rand(len(env_ids), 3, device=self.device) - 1.0
        )
        noise_quat = quat_from_euler_xyz(
            rotation_noise[:, 0], rotation_noise[:, 1], rotation_noise[:, 2]
        )
        object_pose = torch.cat(
            (object_position, quat_mul(noise_quat, self._ref_obj_quat[self._frame[env_ids]])),
            dim=-1,
        )
        object_velocity = torch.cat(
            (
                self._ref_obj_linvel[self._frame[env_ids]],
                self._ref_obj_angvel[self._frame[env_ids]],
            ),
            dim=-1,
        )
        self.object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim_index(
            root_velocity=object_velocity, env_ids=env_ids
        )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._action_delta[env_ids] = 0.0
        self._targets[env_ids] = self._ref_q[self._frame[env_ids]]
        self._wrist_target_quat[env_ids] = self._ref_wrist_quat[self._frame[env_ids]]
        self._compute_state()

    @staticmethod
    def _euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
        return quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

    def _compute_state(self) -> None:
        root_pos = self.robot.data.root_pos_w.torch - self.scene.env_origins
        root_quat = self.robot.data.root_quat_w.torch
        root_vel = self.robot.data.root_vel_w.torch
        finger_q = self.robot.data.joint_pos.torch[:, self._finger_ids]
        finger_qd = self.robot.data.joint_vel.torch[:, self._finger_ids]
        wrist_euler = torch.stack(euler_xyz_from_quat(root_quat), dim=-1)
        self._hand_q = torch.cat((root_pos, wrist_euler, finger_q), dim=-1)
        self._hand_qd = torch.cat((root_vel, finger_qd), dim=-1)
        self._object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        self._object_quat = self.object.data.root_quat_w.torch
        object_velocity = self.object.data.root_vel_w.torch
        self._object_linvel = object_velocity[:, :3]
        self._object_angvel = object_velocity[:, 3:]
