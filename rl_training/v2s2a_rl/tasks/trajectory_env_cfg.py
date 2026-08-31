"""Isaac Lab 3.0 configuration for trajectory-conditioned hand manipulation."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from v2s2a_rl.bundle import TaskBundle
from v2s2a_rl.config import bundle_from_env


@configclass
class TrajectoryTrackingEnvCfg(DirectRLEnvCfg):
    """Residual imitation with an object-assistance curriculum.

    The complete upstream scene MJCF is imported as one articulation. This is
    intentional: it preserves hand collision geometry, reconstructed object
    convex parts, pedestals/supports, mass and contact settings as one asset.
    """

    bundle_path: str = ""
    num_hand_dofs: int = 1  # patched from the bundle in __post_init__
    control_dt: float = 0.02
    wrist_position_action_scale: float = 0.03
    wrist_rotation_action_scale: float = 0.08
    wrist_position_stiffness: float = 300.0
    wrist_velocity_damping: float = 30.0
    wrist_rotation_stiffness: float = 30.0
    wrist_angular_damping: float = 1.0
    wrist_max_force: float = 200.0
    wrist_max_torque: float = 60.0
    finger_action_scale: float = 0.10
    action_ema: float = 0.35
    reference_speed: float = 1.0
    # Reference-state initialization bootstraps local tracking. The curriculum
    # progressively increases frame-0 episodes and removes object assistance.
    reset_frame_fraction: float = 0.8
    reset_first_frame_probability: float = 0.1
    curriculum_initial_difficulty: float = 0.0
    curriculum_final_reset_frame_fraction: float = 0.8
    curriculum_final_first_frame_probability: float = 0.5
    curriculum_min_reset_phase: float = 0.0
    curriculum_min_reset_noise_scale: float = 0.25
    curriculum_success_smoothing: float = 0.95
    curriculum_success_threshold: float = 0.08
    curriculum_failure_threshold: float = 0.02
    curriculum_difficulty_step: float = 0.01
    curriculum_min_completed_episodes: int = 16
    object_assist_initial: float = 1.0
    object_assist_decay_steps: int = 20_000
    object_assist_force: float = 50.0
    object_assist_torque: float = 8.0
    object_assist_position_stiffness: float = 100.0
    object_assist_velocity_damping: float = 10.0
    object_assist_rotation_stiffness: float = 20.0
    object_assist_angular_damping: float = 2.0
    object_position_sigma: float = 0.04
    object_rotation_sigma: float = 0.35
    relative_position_sigma: float = 0.04
    relative_rotation_sigma: float = 0.35
    hand_position_sigma: float = 0.35
    hand_keypoint_sigma: float = 0.03
    contact_distance_threshold: float = 0.025
    contact_force_threshold: float = 0.15
    contact_force_target: float = 2.0
    success_position_tolerance: float = 0.04
    success_rotation_tolerance: float = 0.4
    early_termination_position_error: float = 0.35
    goal_reward_start_phase: float = 0.8
    joint_reset_noise: float = 0.015
    object_reset_position_noise: float = 0.005
    object_reset_rotation_noise: float = 0.03
    # Enable gain randomization only after the nominal frame-0 benchmark works.
    joint_stiffness_scale_range: tuple[float, float] = (1.0, 1.0)
    joint_damping_scale_range: tuple[float, float] = (1.0, 1.0)
    future_command_steps: tuple[int, ...] = (1, 2, 4, 8)

    action_space: int = 1
    observation_space: int = 1
    state_space: int = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=1.0,
        replicate_physics=False,
        clone_in_fabric=True,
    )
    # Convex object/fingertip contacts exceed PhysX's default GPU patch pool at
    # thousands of parallel environments (observed requirement: ~234k at 4096).
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=2,
        physics=PhysxCfg(gpu_max_rigid_patch_count=1_048_576),
    )
    # The task occupies a roughly 0.5 m workspace around the origin. Isaac
    # Lab's 7.5 m default camera makes the hand effectively invisible.
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.34, -0.36, 0.30),
        lookat=(-0.05, 0.0, 0.12),
        origin_type="env",
        env_index=0,
        resolution=(1280, 720),
    )

    scene_asset: ArticulationCfg | None = None
    object_asset: RigidObjectCfg | None = None
    fingertip_contacts: dict[str, ContactSensorCfg] | None = None

    def __post_init__(self) -> None:
        bundle = TaskBundle.from_json(self.bundle_path or bundle_from_env())
        if not bundle.scene_usd_path:
            raise ValueError("task bundle has no converted scene_usd_path; rerun `v2s2a-rl prepare`")
        self.bundle_path = str(bundle_from_env()) if not self.bundle_path else self.bundle_path
        self.num_hand_dofs = bundle.hand_dofs
        self.action_space = bundle.hand_dofs
        # phase + q/qdot errors + object pose/velocity/errors + previous action
        # + future hand/object deltas for each look-ahead step.
        self.observation_space = (
            2
            + 4 * bundle.hand_dofs
            + 3 + 4 + 3 + 3 + 3 + 4
            + 3 + 4 + 3 + 4  # live wrist-object pose and its reference error
            + bundle.hand_dofs
            + len(self.future_command_steps) * (bundle.hand_dofs + 3 + 4)
        )
        self.decimation = max(1, round(self.control_dt / self.sim.dt))
        self.sim.render_interval = self.decimation
        self.episode_length_s = bundle.num_frames / bundle.frequency / self.reference_speed

        self.scene_asset = ArticulationCfg(
            prim_path="/World/envs/env_.*/Scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=bundle.scene_usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=2.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
            # Floating palm root replaces the six Cartesian wrist joints.
            articulation_root_prim_path=f"/Geometry/{bundle.robot_root_body_name}",
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
            actuators={
                "hand": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=80.0,
                    velocity_limit_sim=16.0,
                    stiffness=60.0,
                    damping=3.0,
                    armature=0.001,
                    friction=0.001,
                )
            },
        )
        object_prim = f"/World/envs/env_.*/Scene/Geometry/{bundle.object_body_name}"
        self.object_asset = RigidObjectCfg(
            prim_path=object_prim,
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        self.fingertip_contacts = {
            name: ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Scene/Geometry/{path}",
                filter_prim_paths_expr=[object_prim],
                history_length=0,
                track_contact_points=False,
                track_pose=False,
                force_threshold=self.contact_force_threshold,
                max_contact_data_count_per_prim=128,
            )
            for name, path in zip(
                bundle.fingertip_body_names, bundle.fingertip_body_paths, strict=True
            )
        }


@configclass
class TrajectoryTrackingPlayEnvCfg(TrajectoryTrackingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.reset_frame_fraction = 0.0
        self.curriculum_final_reset_frame_fraction = 0.0
        self.reset_first_frame_probability = 1.0
        self.curriculum_final_first_frame_probability = 1.0
        self.object_assist_initial = 0.0
