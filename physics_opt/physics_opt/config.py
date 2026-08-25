"""Define the configuration for the optimizer."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, fields

import loguru
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from physics_opt.paths import PROJECT_ROOT
from physics_opt.utils.io import get_processed_data_dir, resolve_auto_embodiment


@dataclass
class Config:
    # === TASK CONFIGURATION ===
    robot_type: str = "sharpa"
    embodiment_type: str = "bimanual"  # "auto", "left", "right", "bimanual"
    task: str = ""
    seed: int = 0

    # === DATASET CONFIGURATION ===
    output_root_dir: str = str(PROJECT_ROOT / "outputs")
    dataset_name: str = "do_as_i_do"
    data_id: int = 0
    model_path: str = ""
    data_path: str = ""

    # === SIMULATOR CONFIGURATION ===
    simulator: str = "mjwp"
    device: str = "cuda:0"
    # Simulation timing
    sim_dt: float = 0.01
    ctrl_dt: float = 0.4
    ref_dt: float = 0.02
    render_dt: float = 0.02
    horizon: float = 1.6  # planning horizon
    knot_dt: float = 0.4  # knot point spacing
    max_sim_steps: int = -1  # -1 for unlimited
    blend_reference: bool = (
        True  # blend appended IK reference with optimized tail for smooth warm starts
    )
    # Simulation constraints
    nconmax_per_env: int = 100
    njmax_per_env: int = 500
    # Simulation annealing
    num_dyn: int = (
        1  # number of environments for annealing, used for virtual contact constraint
    )
    # Domain randomization
    num_dr: int = (
        1  # number of domain randomization groups, used for domain randomization
    )
    pair_margin_range: tuple[float, float] = (-0.005, 0.005)
    xy_offset_range: tuple[float, float] = (-0.005, 0.005)
    perturb_force: float = 0.0
    perturb_torque: float = 0.0
    perturb_force_scale: float = (
        0.0  # random per-world force as fraction of gravity (force = scale * mass * g)
    )
    perturb_torque_scale: float = (
        0.0  # random per-world torque on objects each step (0 = disabled)
    )
    perturb_prob: float = (
        0.0  # probability of starting a new perturbation at each sim step
    )
    perturb_continue_prob: float = (
        0.0  # probability that an active perturbation continues to the next step
    )
    # Lag eroding both edges of the perturbation gate's "held and lifted off any
    # rest surface" signal: gate[t] is forced False unless that signal held for
    # the whole window [t - perturb_gate_lag_steps + 1, t + perturb_gate_lag_steps
    # - 1]. Delays disturbances until the grasp has settled AND ends them before
    # release, so perturbation only fires well inside a stable airborne hold.
    # See ``erode_mask``. Note this lag shapes only the perturbation gate, not
    # the raw in-hand signal the rest/floor-mismatch penalties read.
    perturb_gate_lag_duration: float = 0.5  # seconds (0 = no lag)
    perturb_gate_lag_steps: int = 0  # derived from perturb_gate_lag_duration / sim_dt
    # Number of perturbation seeds K to evaluate per control candidate. Each
    # candidate is replicated K times (total sim worlds = num_samples * K) and
    # the per-candidate reward is averaged over its K replicas. Within a step,
    # all candidates share the same K random draws (common-random-numbers), so
    # reward differences between candidates reflect control quality rather
    # than disturbance luck. The contact gate remains per-world.
    num_perturb_samples: int = 1
    contact_guidance: bool = False
    object_pos_actuator_names: list[str] = field(
        default_factory=lambda: [
            "right_object_pos_x",
            "right_object_pos_y",
            "right_object_pos_z",
            "left_object_pos_x",
            "left_object_pos_y",
            "left_object_pos_z",
        ]
    )
    object_rot_actuator_names: list[str] = field(
        default_factory=lambda: [
            "right_object_rot_x",
            "right_object_rot_y",
            "right_object_rot_z",
            "left_object_rot_x",
            "left_object_rot_y",
            "left_object_rot_z",
        ]
    )
    object_action_dims: int = 0
    object_actuator_ids: list[int] = field(default_factory=list)
    object_actuator_names: list[str] = field(default_factory=list)
    init_pos_actuator_gain: float = 10.0
    init_pos_actuator_bias: float = 10.0
    init_rot_actuator_gain: float = 0.1
    init_rot_actuator_bias: float = 0.1
    guidance_decay_ratio: float = 0.5
    gibbs_sampling: bool = False
    # Warmup: prepend static copies of frame 0 with weld constraint holding object
    warmup_duration: float = 0.0  # warmup duration in seconds (0 = disabled)
    warmup_steps: int = 0  # derived from warmup_duration / sim_dt
    warmup_weld_eq_ids: list[int] = field(default_factory=list)
    warmup_sim_step: int = 0  # current absolute sim step (set before optimize)
    warmup_rew_multiplier: float = (
        1.0  # scale reward during warmup (0=ignore, 1=normal)
    )
    default_gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    warmup_ref_base_interp_duration: float = 0.0  # seconds to interpolate base (wrist+object) ref from init to frame 0 (0 = disabled)
    warmup_ref_base_interp_steps: int = (
        0  # derived from warmup_ref_base_interp_duration / sim_dt
    )
    warmup_ref_finger_interp_duration: float = (
        0.0  # seconds to interpolate finger ref from init to frame 0 (0 = disabled)
    )
    warmup_ref_finger_interp_steps: int = (
        0  # derived from warmup_ref_finger_interp_duration / sim_dt
    )
    warmup_min_clearance: float = 0.0  # extra distance added past the first penetration-free init pose (meters, 0 = disabled)

    # === OPTIMIZER CONFIGURATION ===
    # Sampling parameters
    num_samples: int = 2048
    temperature: float = 0.3
    max_num_iterations: int = 16
    improvement_threshold: float = 0.01
    improvement_check_steps: int = 1
    # Termination parameters
    terminate_z_threshold: float = -1.0
    terminate_penetration_threshold: float = 1.0e9
    # Compilation
    use_torch_compile: bool = True
    # Noise scheduling
    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    final_noise_scale: float = 0.1
    exploit_ratio: float = 0.01
    exploit_noise_scale: float = 0.01
    zero_first_knot_noise: bool = (
        True  # zero noise at first knot for smooth execution boundaries
    )
    # Noise scaling by component
    joint_noise_scale: float = 0.15
    pos_noise_scale: float = 0.03
    rot_noise_scale: float = 0.03
    # When True, joint_noise_scale is interpreted as a fraction of each joint's
    # actuator ctrlrange rather than an absolute stddev (rad). Equalizes the
    # explored fraction of range across joints with very different limits
    fractional_joint_noise: bool = False
    # Reward scaling
    base_pos_rew_scale: float = 1.0
    base_rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.003
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    vel_rew_scale: float = 0.0001
    terminal_rew_scale: float = 1.0
    contact_rew_scale: float = 0.0
    penetration_penalty_scale: float = 0.0  # penalty per meter of max penetration depth
    penetration_margin: float = (
        0.003  # allow this much penetration (meters) before penalty kicks in
    )
    drop_penalty_scale: float = 0.0  # penalty when object z falls below drop_z_thresh
    drop_z_thresh: float = 0.0  # z height below which drop penalty activates
    # Penalty for object/pedestal contact mismatching the per-frame in-hand gate
    # (same gate used by random perturbation). On frames where the object is
    # supposed to be in-hand, penalize any pedestal contact; on frames where it
    # is supposed to rest, penalize the absence of pedestal contact. Per side,
    # per world, binary (0 or 1) before scaling. 0 disables.
    pedestal_penalty_scale: float = 0.0
    # Distance threshold (meters) for the reference object-to-pedestal
    # proximity check that gates the pedestal-mismatch penalty. The in-hand
    # branch only fires on frames where the reference object is *farther* than
    # this from every pedestal; the rest branch only on frames where it is
    # *closer*. Metric is the object's lowest vertex height above a pedestal's
    # top face (over its footprint) — the pedestal analog of
    # ``object_floor_distance_thresh``, collapsing to ~0 at rest so small
    # thresholds behave intuitively. See ``compute_near_pedestal_mask``.
    object_pedestal_distance_thresh: float = 0.01
    # In-hand threshold (meters) used by the post-IK pedestal-placement step
    # to decide whether an endpoint is in-hand (no pedestal) or at rest
    # (place a pedestal under the object). Min vertex distance from the hand
    # surface to the object mesh. See ``retargeting/pipeline/resolve_pedestal.py``.
    hand_object_distance_thresh: float = 0.1
    # Floor analog of ``pedestal_penalty_scale`` for datasets where the object
    # rests directly on the floor (object_floor_collision=True) rather than on a
    # pedestal. On frames where the object is supposed to be in-hand, penalize
    # the absence of hand contact; on frames where it is supposed to rest on the
    # floor, penalize the absence of object-floor contact. Per side, per world,
    # binary (0 or 1) before scaling. 0 disables.
    floor_penalty_scale: float = 0.0
    # Distance threshold (meters) for the reference object-to-floor proximity
    # check that gates the floor-mismatch penalty. Metric is the object's lowest
    # world-frame vertex height above the floor plane. Smaller than
    # ``object_pedestal_distance_thresh`` because floor-rest reference data is
    # assumed accurate.
    object_floor_distance_thresh: float = 0.01

    # === VISUALIZATION CONFIGURATION ===
    show_viewer: bool = True
    viewer: str = "viser"
    wait_on_finish: bool = True  # block after optimization to keep viewer alive
    save_video: bool = False
    save_info: bool = True
    save_metrics: bool = True
    save_config: bool = True
    force: bool = False
    # Subdirectory under the trial dir for stage-5 artifacts (empty = beside
    # the IK inputs). Keeps each pipeline stage in its own folder under the
    # clip root.
    output_subdir: str = ""

    # === TRACE RECORDING ===
    trace_dt: float = 1 / 50.0
    num_trace_uniform_samples: int = 12
    num_trace_topk_samples: int = 4
    trace_site_ids: list = field(default_factory=list)
    num_object_trace_sites: int = 0

    # === CONTACT GUIDANCE (DERIVED) ===
    contact_order: list = field(default_factory=list)
    hand_contact_site_ids: list = field(default_factory=list)
    # Object-side contact site ids (populated from task_info.json when
    # contact_rew_scale > 0; see optimize_physics setup).
    contact_site_ids: list = field(default_factory=list)
    right_contact_indices: list = field(default_factory=list)
    left_contact_indices: list = field(default_factory=list)
    right_pos_ctrl_ids: list = field(default_factory=list)
    left_pos_ctrl_ids: list = field(default_factory=list)
    contact_len: int = 0

    # === AUTOMATICALLY SET PROPERTIES ===
    # Computed timesteps
    horizon_steps: int = -1
    knot_steps: int = -1  # sim-steps per joint knot
    ref_steps: int = -1
    ctrl_steps: int = -1
    # Model dimensions
    nq_obj: int = -1  # object DOF
    nq: int = -1  # total position DOF
    nv: int = -1  # total velocity DOF
    nu: int = -1  # total control DOF
    npair: int = -1  # total pair DOF
    # Computed tensors
    noise_scale: torch.Tensor = field(default_factory=lambda: torch.ones(1))
    # Per-actuator ctrlrange size (hi - lo), shape (nu,). Populated from the
    # MuJoCo model when the simulator is available. Used by
    # fractional_joint_noise.
    actuator_ranges: torch.Tensor = field(default_factory=lambda: torch.ones(1))
    beta_traj: float = -1.0
    # Runtime state
    env_params_list: list = field(default_factory=list)
    viewer_body_entity_and_ids: list = field(default_factory=list)
    output_dir: str = ""


def resolve_object_actuator_ids(
    model: mujoco.MjModel,
    desired_names: list[str],
    object_action_dims: int,
) -> tuple[list[int], list[str]]:
    """Resolve object actuator ids by name, with a fallback to last N actuators."""
    resolved_ids: list[int] = []
    resolved_names: list[str] = []
    for name in desired_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid != -1:
            resolved_ids.append(int(aid))
            resolved_names.append(name)
    if resolved_ids:
        if len(resolved_ids) != len(desired_names):
            loguru.logger.info(
                "Resolved {} / {} object actuators by name.",
                len(resolved_ids),
                len(desired_names),
            )
        return resolved_ids, resolved_names

    obj_start = max(model.nu - max(object_action_dims, 0), 0)
    fallback_ids = list(range(obj_start, model.nu))
    fallback_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
        for aid in fallback_ids
    ]
    loguru.logger.warning(
        "No named object actuators found; falling back to last {} actuators.",
        len(fallback_ids),
    )
    return fallback_ids, fallback_names


def load_config_yaml(path: str) -> dict:
    """Load a config YAML into a plain dict, resolving OmegaConf types."""
    if not path:
        return {}
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    cfg = OmegaConf.load(abs_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Config at {abs_path} did not resolve to a mapping.")
    return cfg_dict


def filter_config_fields(config_dict: dict) -> dict:
    allowed = {field.name for field in fields(Config)}
    return {key: value for key, value in config_dict.items() if key in allowed}


def build_hand_contact_site_ids(
    mj_model: mujoco.MjModel, embodiment_type: str
) -> tuple[list[tuple[str, str]], list[int | None]]:
    contact_order = []
    if embodiment_type in ["bimanual", "right"]:
        contact_order.extend(
            [
                ("right", "thumb"),
                ("right", "index"),
                ("right", "middle"),
                ("right", "ring"),
                ("right", "pinky"),
            ]
        )
    if embodiment_type in ["bimanual", "left"]:
        contact_order.extend(
            [
                ("left", "thumb"),
                ("left", "index"),
                ("left", "middle"),
                ("left", "ring"),
                ("left", "pinky"),
            ]
        )

    site_ids: list[int | None] = [None] * len(contact_order)
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is None:
            continue
        name_l = name.lower()
        if "track" not in name_l or "hand" not in name_l:
            continue
        for idx, (side, finger) in enumerate(contact_order):
            if side in name_l and finger in name_l:
                if site_ids[idx] is None:
                    site_ids[idx] = sid
                break

    missing = [i for i, sid in enumerate(site_ids) if sid is None]
    if missing:
        loguru.logger.warning(
            "Missing {} hand contact sites for guidance; indices: {}",
            len(missing),
            missing,
        )
    return contact_order, site_ids


def get_object_pos_ctrl_indices(config: Config) -> tuple[list[int], list[int]]:
    right_ids: list[int] = []
    left_ids: list[int] = []
    if config.object_actuator_ids and config.object_actuator_names:
        for aid, name in zip(
            config.object_actuator_ids, config.object_actuator_names, strict=False
        ):
            name_l = (name or "").lower()
            if "_pos_" not in name_l:
                continue
            if "right" in name_l:
                right_ids.append(int(aid))
            elif "left" in name_l:
                left_ids.append(int(aid))

    if right_ids or left_ids:
        return right_ids, left_ids

    obj_dims = int(config.object_action_dims) if config.object_action_dims > 0 else 0
    if obj_dims == 0:
        obj_dims = 12 if config.embodiment_type == "bimanual" else 6
    start = max(int(config.nu) - obj_dims, 0)
    if config.embodiment_type == "bimanual" and obj_dims >= 12:
        right_ids = list(range(start, start + 3))
        left_ids = list(range(start + 3, start + 6))
    else:
        if config.embodiment_type == "right":
            right_ids = list(range(start, start + 3))
        elif config.embodiment_type == "left":
            left_ids = list(range(start, start + 3))
        else:
            right_ids = list(range(start, start + 3))
    return right_ids, left_ids


def _build_logspace_ramp(config: Config, n_knots: int) -> torch.Tensor:
    """Logspace noise ramp, shape (1, n_knots, 1) to broadcast over (samples, knots, dims)."""
    return torch.logspace(
        start=torch.log10(torch.tensor(config.first_ctrl_noise_scale)),
        end=torch.log10(torch.tensor(config.last_ctrl_noise_scale)),
        steps=n_knots,
        device=config.device,
        base=10,
    )[None, :, None]


def _finalize_noise_tensor(config: Config, ns: torch.Tensor) -> torch.Tensor:
    """Zero-first-knot, repeat to num_samples, zero sample 0, exploit-scale the tail.

    ns: (1, n_knots, dim) -> (num_samples, n_knots, dim).
    """
    if config.zero_first_knot_noise:
        ns = ns.clone()
        ns[:, 0, :] = 0.0
    ns = ns.repeat(config.num_samples, 1, 1)
    ns[0] *= 0.0
    num_exploit_samples = int(config.num_samples * config.exploit_ratio)
    if num_exploit_samples > 0:
        ns[-num_exploit_samples:] *= config.exploit_noise_scale
    return ns


def get_noise_scale(config: Config) -> torch.Tensor:
    """Noise tensor (num_samples, n_joint_knots, nu) covering all ctrl dims:
    wrist/object via pos/rot_noise_scale, joints via joint_noise_scale.
    """
    n_joint_knots = int(round(config.horizon / config.knot_dt))
    base_logspace = _build_logspace_ramp(config, n_joint_knots)  # (1, n_joint_knots, 1)

    noise_scale = base_logspace.repeat(1, 1, config.nu)
    if config.fractional_joint_noise:
        joint_mult = config.joint_noise_scale * config.actuator_ranges.to(
            device=config.device
        )
    else:
        joint_mult = torch.full(
            (config.nu,), config.joint_noise_scale, device=config.device
        )
    if config.embodiment_type in ["bimanual", "right", "left"]:
        object_action_dims = max(int(config.object_action_dims), 0)
        robot_nu = max(int(config.nu - object_action_dims), 0)
        noise_scale[:, :, :3] *= config.pos_noise_scale
        noise_scale[:, :, 3:6] *= config.rot_noise_scale
        if config.embodiment_type == "bimanual":
            half_dof = robot_nu // 2
            noise_scale[:, :, 6:half_dof] *= joint_mult[6:half_dof]
            noise_scale[:, :, half_dof : half_dof + 3] *= config.pos_noise_scale
            noise_scale[:, :, half_dof + 3 : half_dof + 6] *= config.rot_noise_scale
            noise_scale[:, :, half_dof + 6 : robot_nu] *= joint_mult[
                half_dof + 6 : robot_nu
            ]
        elif config.embodiment_type in ["right", "left"]:
            noise_scale[:, :, 6:robot_nu] *= joint_mult[6:robot_nu]
    else:
        noise_scale *= joint_mult
    if config.contact_guidance and config.object_actuator_ids:
        object_ids = torch.as_tensor(
            config.object_actuator_ids, device=config.device, dtype=torch.long
        )
        noise_scale[:, :, object_ids] *= 0.0

    return _finalize_noise_tensor(config, noise_scale)


def compute_steps(config: Config):
    # make sure every dt can be divided by sim_dt
    config.horizon_steps = int(np.round(config.horizon / config.sim_dt))
    config.knot_steps = int(np.round(config.knot_dt / config.sim_dt))
    config.ref_steps = int(np.round(config.ref_dt / config.sim_dt))
    config.ctrl_steps = int(np.round(config.ctrl_dt / config.sim_dt))
    assert np.isclose(
        config.horizon - config.horizon_steps * config.sim_dt, 0, atol=1e-5
    ), "horizon must be divisible by sim_dt"
    assert np.isclose(
        config.ctrl_dt - config.ctrl_steps * config.sim_dt, 0, atol=1e-5
    ), "ctrl_dt must be divisible by sim_dt"
    assert np.isclose(
        config.knot_dt - config.knot_steps * config.sim_dt, 0, atol=1e-5
    ), "knot_dt must be divisible by sim_dt"
    config.warmup_steps = int(np.round(config.warmup_duration / config.sim_dt))
    config.warmup_ref_base_interp_steps = int(
        np.round(config.warmup_ref_base_interp_duration / config.sim_dt)
    )
    config.warmup_ref_finger_interp_steps = int(
        np.round(config.warmup_ref_finger_interp_duration / config.sim_dt)
    )
    config.perturb_gate_lag_steps = int(
        np.round(config.perturb_gate_lag_duration / config.sim_dt)
    )
    return config


def compute_noise_schedule(config: Config) -> Config:
    config.noise_scale = get_noise_scale(config)
    if config.max_num_iterations > 0:
        config.beta_traj = config.final_noise_scale ** (1 / config.max_num_iterations)
    else:
        config.beta_traj = 1.0
    return config


def process_config(config: Config):
    if config.embodiment_type == "auto":
        config.embodiment_type = resolve_auto_embodiment(
            config.dataset_name, config.output_root_dir, config.task
        )

    config = compute_steps(config)
    trace_steps_tmp = int(np.round(config.trace_dt / config.sim_dt))
    assert np.isclose(
        config.trace_dt - trace_steps_tmp * config.sim_dt, 0, atol=1e-3
    ), "trace_dt must be divisible by sim_dt"

    needs_act = config.contact_guidance
    if needs_act:
        config.nq_obj = {
            "bimanual": 12,
            "right": 6,
            "left": 6,
        }.get(config.embodiment_type, 0)
    else:
        config.nq_obj = {
            "bimanual": 14,
            "right": 7,
            "left": 7,
        }.get(config.embodiment_type, 0)

    output_root_dir_abs = os.path.abspath(config.output_root_dir)
    processed_dir_robot = get_processed_data_dir(
        output_root_dir=output_root_dir_abs,
        dataset_name=config.dataset_name,
        robot_type=config.robot_type,
        embodiment_type=config.embodiment_type,
        task=config.task,
        data_id=config.data_id,
    )
    # scene_eq.xml supports annealing over equality constraints
    needs_act = config.contact_guidance
    if needs_act:
        scene_xml = "scene_act.xml"
    else:
        scene_xml = "scene.xml" if config.num_dyn == 1 else "scene_eq.xml"
    config.model_path = f"{processed_dir_robot}/{scene_xml}"
    # default to MJWP retargeted trajectory if available
    if needs_act:
        config.data_path = f"{processed_dir_robot}/trajectory_kinematic_act.npz"
    else:
        config.data_path = f"{processed_dir_robot}/trajectory_kinematic.npz"

    if config.simulator == "mjwp":
        model = mujoco.MjModel.from_xml_path(config.model_path)
        config.nq = model.nq
        config.nv = model.nv
        config.nu = model.nu
        config.npair = model.npair
        ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
        config.actuator_ranges = torch.as_tensor(
            ctrlrange[:, 1] - ctrlrange[:, 0], device=config.device
        )
        if needs_act:
            if config.object_action_dims <= 0:
                config.object_action_dims = (
                    12 if config.embodiment_type == "bimanual" else 6
                )
            desired_names = (
                config.object_pos_actuator_names + config.object_rot_actuator_names
            )
            object_ids, object_names = resolve_object_actuator_ids(
                model,
                desired_names,
                config.object_action_dims,
            )
            config.object_actuator_ids = object_ids
            config.object_actuator_names = object_names
            config.right_pos_ctrl_ids, config.left_pos_ctrl_ids = (
                get_object_pos_ctrl_indices(config)
            )
            config.contact_order, config.hand_contact_site_ids = (
                build_hand_contact_site_ids(model, config.embodiment_type)
            )
            config.right_contact_indices = [
                idx
                for idx, (side, finger) in enumerate(config.contact_order)
                if (side == "right") and (finger in ["thumb"])
            ]
            config.left_contact_indices = [
                idx
                for idx, (side, finger) in enumerate(config.contact_order)
                if side == "left" and (finger in ["thumb"])
            ]

    if config.warmup_steps > 0 and config.simulator == "mjwp":
        weld_ids = []
        for side in ("right", "left"):
            eq_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{side}_object_weld"
            )
            if eq_id != -1:
                weld_ids.append(eq_id)
        config.warmup_weld_eq_ids = weld_ids
        if not weld_ids:
            loguru.logger.warning(
                "warmup_steps > 0 but no weld constraints found; regenerate scene XML."
            )

    config = compute_noise_schedule(config)

    # write artifacts alongside the trial (or in a stage-5 subfolder of it)
    config.output_dir = (
        os.path.join(processed_dir_robot, config.output_subdir)
        if config.output_subdir
        else processed_dir_robot
    )
    os.makedirs(config.output_dir, exist_ok=True)

    task_info_path = f"{processed_dir_robot}/../task_info.json"
    try:
        with open(task_info_path, encoding="utf-8") as f:
            task_info = json.load(f)
    except FileNotFoundError:
        loguru.logger.warning(
            f"task_info.json not found at {task_info_path}, using default values"
        )
        task_info = {}
    if "ref_dt" in task_info:
        config.ref_dt = task_info["ref_dt"]
        loguru.logger.info(f"overriding ref_dt: {config.ref_dt} from task_info.json")

    if config.contact_rew_scale > 0.0:
        if "contact_site_ids" in task_info:
            config.contact_site_ids = task_info["contact_site_ids"]
            loguru.logger.info(
                f"overriding contact_site_ids: {config.contact_site_ids} from task_info.json"
            )
        else:
            raise ValueError(
                "contact_site_ids not found in task_info.json while contact_rew_scale > 0.0"
            )

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)

    return config
