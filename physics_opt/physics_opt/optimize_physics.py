"""Sampling-based MPC physics optimization with MuJoCo Warp."""

from __future__ import annotations

import time
from dataclasses import fields
from pathlib import Path

import imageio
import loguru
import mujoco
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

from physics_opt.config import (
    Config,
    process_config,
)
from physics_opt.utils.interp import get_slice
from physics_opt.utils.io import (
    load_data,
    warmup_ref_interp,
)
from physics_opt.utils.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from physics_opt.utils.tracking_error import compute_object_tracking_error
from physics_opt.utils.mjwp import (
    check_penetration,
    compute_contact_point_delta,
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    precompute_hand_object_geom_mask,
    save_env_params,
    save_state,
    set_gravity,
    set_weld_active,
    set_weld_target,
    setup_env,
    setup_mj_model,  # mjwp specific
    step_env,
    sync_env,
    warmup_analytical_init,
)
from physics_opt.utils.viewer import (
    REF_COLOR_BLUE,
    REF_COLOR_RED,
    log_frame,
    render_image,
    setup_renderer,
    setup_viewer,
    update_viewer,
)
from physics_opt.utils import viser_viewer as _viser_viewer

_CONFIG_SKIP_FIELDS = {
    "noise_scale",
    "env_params_list",
    "viewer_body_entity_and_ids",
}


def _assert_object_actuator_gains_zero(
    env, config: Config, stage: str, atol: float = 1e-4
) -> None:
    if not config.contact_guidance or not config.object_actuator_ids:
        return
    actuator_ids = np.asarray(config.object_actuator_ids, dtype=int)
    if not hasattr(env, "model_wp") or not hasattr(env.model_wp, "actuator_gainprm"):
        raise AssertionError("MJWarp model does not expose actuator_gainprm.")
    gainprm = wp.to_torch(env.model_wp.actuator_gainprm).detach().cpu().numpy()
    biasprm = wp.to_torch(env.model_wp.actuator_biasprm).detach().cpu().numpy()
    if gainprm.ndim == 3:
        gainprm = gainprm[0]
    if biasprm.ndim == 3:
        biasprm = biasprm[0]
    kp = gainprm[actuator_ids, 0]
    kd = -biasprm[actuator_ids, 1]
    assert np.allclose(kp, 0.0, atol=atol), (
        f"Object actuator Kp not near zero at {stage}: max={np.max(np.abs(kp))}"
    )
    assert np.allclose(kd, 0.0, atol=atol), (
        f"Object actuator Kd not near zero at {stage}: max={np.max(np.abs(kd))}"
    )


def _normalize_yaml_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _save_config_yaml(config: Config) -> None:
    if not config.save_config:
        return
    config_dict = {}
    for field in fields(config):
        if field.name in _CONFIG_SKIP_FIELDS:
            continue
        config_dict[field.name] = _normalize_yaml_value(getattr(config, field.name))
    output_path = (
        Path(config.output_dir)
        / f"config{'_act' if config.contact_guidance else ''}.yaml"
    )
    OmegaConf.save(config=OmegaConf.create(config_dict), f=str(output_path))
    loguru.logger.info(f"Saved config to {output_path}")


def _get_bimanual_hand_indices(config: Config) -> tuple[list[int], list[int]]:
    robot_nu = int(config.nu)
    if config.contact_guidance:
        obj_dims = (
            int(config.object_action_dims) if config.object_action_dims > 0 else 12
        )
        robot_nu = max(robot_nu - obj_dims, 0)
    half = robot_nu // 2
    right_ids = list(range(0, half))
    left_ids = list(range(half, robot_nu))
    return right_ids, left_ids


def _apply_noise_mask(
    base_noise_scale: torch.Tensor, zero_indices: list[int]
) -> torch.Tensor:
    noise_scale = base_noise_scale.clone()
    if zero_indices:
        idx = torch.as_tensor(
            zero_indices, device=base_noise_scale.device, dtype=torch.long
        )
        noise_scale[:, :, idx] *= 0.0
    return noise_scale


def _gate_state_lane(env, bid: int, warmup_steps: int) -> np.ndarray:
    """Per-frame gate state for object body ``bid``, as (T,) int8: 1=held
    (in-hand, far from rest surface), 2=at rest (near pedestal/floor, not
    in-hand), 0=neither/warmup.

    Mirrors the penalty branches in mjwp (_rest_surface_penalty /
    _ref_contact_state), which read the raw un-lagged in-hand mask — NOT the
    perturbation gate, whose extra erosion lag would not match. Warmup frames
    are forced to 0 (the masks themselves are not warmup-suppressed).
    """
    in_hand = env._inhand_gate_mask[bid].detach().cpu().numpy().astype(bool)
    near_rest = np.zeros_like(in_hand)
    for d in (env._obj_pedestal_gate_mask, env._obj_floor_gate_mask):
        m = d.get(bid)
        if m is not None and m.numel() == in_hand.size:
            near_rest |= m.detach().cpu().numpy().astype(bool)
    state = np.zeros(in_hand.shape, dtype=np.int8)
    state[in_hand & ~near_rest] = 1
    state[near_rest & ~in_hand] = 2
    if warmup_steps > 0:
        state[: int(warmup_steps)] = 0
    return state


def main(config: Config):
    config = process_config(config)

    out_path = (
        Path(config.output_dir)
        / f"trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz"
    )
    if not config.force and out_path.exists():
        loguru.logger.info(f"Skipping optimize_physics.py (output exists: {out_path})")
        return
    if config.contact_guidance and config.improvement_threshold > 0.0:
        loguru.logger.warning(
            "contact_guidance requires improvement_threshold <= 0; overriding to 0.0."
        )
        config.improvement_threshold = 0.0

    # load reference data (already interpolated and extended)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
    if (
        config.contact_guidance
        and ctrl_ref.shape[1] != config.nu
        and qpos_ref.shape[1] >= config.nu
    ):
        loguru.logger.info(
            "Using qpos as ctrl reference (ctrl dims: {} -> {}).",
            ctrl_ref.shape[1],
            config.nu,
        )
        ctrl_ref = qpos_ref[:, : config.nu]
    if config.contact_guidance and torch.all(contact <= 0):
        raise ValueError("contact_guidance is enabled, but contact mask is all zeros.")
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    config.max_sim_steps = (
        config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps
    )

    # For warmup, start the simulation with an open hand (finger joints = 0)
    # so the optimizer can close the fingers onto the object surface.
    # The reference stays as the closed grasp — only the initial sim state changes.
    _warmup_init_enabled = False
    finger_qpos_indices: list[int] = []
    if config.warmup_steps > 0 and config.embodiment_type in [
        "bimanual",
        "right",
        "left",
    ]:
        _warmup_init_enabled = True
        nq_obj = config.nq_obj
        robot_nq = qpos_ref.shape[1] - nq_obj
        wrist_dof = 6
        if config.embodiment_type == "bimanual":
            half = robot_nq // 2
            finger_qpos_indices += list(range(wrist_dof, half))
            finger_qpos_indices += list(range(half + wrist_dof, robot_nq))
        else:
            finger_qpos_indices += list(range(wrist_dof, robot_nq))

        # Base open-hand pose; analytical wrist placement runs on GPU after env setup.
        init_qpos = qpos_ref[0].clone()
        init_ctrl = ctrl_ref[0].clone()
        init_qpos[finger_qpos_indices] = 0.0
        init_ctrl[finger_qpos_indices] = 0.0

        # Replace only the first frame for setup_env; reference is unchanged
        init_ref_data = (
            torch.cat([init_qpos.unsqueeze(0), qpos_ref[1:]], dim=0),
            qvel_ref,
            torch.cat([init_ctrl.unsqueeze(0), ctrl_ref[1:]], dim=0),
            contact,
            contact_pos,
        )
        loguru.logger.info(
            "Warmup: set initial hand to open ({} finger joints zeroed).",
            len(finger_qpos_indices),
        )

    else:
        init_ref_data = ref_data

    env = setup_env(config, init_ref_data)
    precompute_hand_object_geom_mask(config, env.model_cpu)

    # GPU warmup: analytical init pose — translate wrist along -palm_normal
    # until penetration-free + palm-facing-object, then add clearance.
    if _warmup_init_enabled:
        warmup_analytical_init(
            config,
            env,
            qpos_ref[0],
            ctrl_ref[0],
            finger_qpos_indices,
        )

    # Interpolate reference from the actual init pose (after analytical placement)
    # to the closed-grasp frame 0. Wrist (base) and finger DOFs can have
    # different interpolation durations to prevent targeting a penetrating
    # reference when the hand is still far from the object. Object dims are
    # never interpolated — they stay at frame 0 throughout warmup.
    base_interp_n = (
        min(config.warmup_ref_base_interp_steps, config.warmup_steps)
        if config.warmup_ref_base_interp_steps > 0
        else 0
    )
    finger_interp_n = (
        min(config.warmup_ref_finger_interp_steps, config.warmup_steps)
        if config.warmup_ref_finger_interp_steps > 0
        else 0
    )
    if (base_interp_n > 0 or finger_interp_n > 0) and config.warmup_steps > 0:
        max_interp_n = max(base_interp_n, finger_interp_n)
        # Read the actual selected init pose from the sim (after analytical placement)
        selected_qpos = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
        selected_ctrl = wp.to_torch(env.data_wp.ctrl)[0].detach().cpu().numpy()
        # Start with the original reference and overwrite interpolated segments
        interped_qpos = qpos_ref[:max_interp_n].detach().cpu().numpy().copy()
        interped_ctrl = ctrl_ref[:max_interp_n].detach().cpu().numpy().copy()
        finger_set = set(finger_qpos_indices)
        # Exclude object dims (last nq_obj) from base — object reference stays static.
        base_qpos_idx = np.array([i for i in range(robot_nq) if i not in finger_set])
        base_ctrl_idx = np.array(
            [
                i
                for i in range(robot_nq)
                if i not in finger_set and i < selected_ctrl.shape[0]
            ]
        )
        finger_idx = (
            np.array(finger_qpos_indices)
            if finger_qpos_indices
            else np.array([], dtype=int)
        )
        if base_interp_n > 0 and len(base_qpos_idx) > 0:
            base_target_qpos = qpos_ref[base_interp_n].detach().cpu().numpy()
            base_target_ctrl = ctrl_ref[base_interp_n].detach().cpu().numpy()
            base_qpos = warmup_ref_interp(
                selected_qpos, base_target_qpos, base_interp_n
            )
            base_ctrl = warmup_ref_interp(
                selected_ctrl, base_target_ctrl, base_interp_n
            )
            interped_qpos[:base_interp_n, base_qpos_idx] = base_qpos[:, base_qpos_idx]
            interped_ctrl[:base_interp_n, base_ctrl_idx] = base_ctrl[:, base_ctrl_idx]
        if finger_interp_n > 0 and len(finger_idx) > 0:
            finger_target_qpos = qpos_ref[finger_interp_n].detach().cpu().numpy()
            finger_target_ctrl = ctrl_ref[finger_interp_n].detach().cpu().numpy()
            fing_qpos = warmup_ref_interp(
                selected_qpos, finger_target_qpos, finger_interp_n
            )
            fing_ctrl = warmup_ref_interp(
                selected_ctrl, finger_target_ctrl, finger_interp_n
            )
            interped_qpos[:finger_interp_n, finger_idx] = fing_qpos[:, finger_idx]
            interped_ctrl[:finger_interp_n, finger_idx] = fing_ctrl[:, finger_idx]
        qpos_ref[:max_interp_n] = torch.from_numpy(interped_qpos).to(qpos_ref)
        ctrl_ref[:max_interp_n] = torch.from_numpy(interped_ctrl).to(ctrl_ref)
        loguru.logger.info(
            "Warmup: interpolating reference — base over {} steps ({:.3f}s), fingers over {} steps ({:.3f}s).",
            base_interp_n,
            base_interp_n * config.sim_dt,
            finger_interp_n,
            finger_interp_n * config.sim_dt,
        )

    # setup mujoco (for viewer only)
    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)

    if _warmup_init_enabled:
        # Show the analytically-placed init pose (all worlds are identical)
        mj_data.qpos[:] = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
        mj_data.ctrl[:] = wp.to_torch(env.data_wp.ctrl)[0].detach().cpu().numpy()
    else:
        mj_data.qpos[:] = init_ref_data[0][0].detach().cpu().numpy()
        mj_data.ctrl[:] = init_ref_data[2][0].detach().cpu().numpy()
    mj_data.qvel[:] = init_ref_data[1][0].detach().cpu().numpy()
    mujoco.mj_forward(mj_model, mj_data)
    mj_data.time = 0.0
    mj_data_ref.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mujoco.mj_forward(mj_model, mj_data_ref)
    warmup_enabled = config.warmup_steps > 0 and len(config.warmup_weld_eq_ids) > 0
    contact_guidance_enabled = (
        config.contact_guidance and len(config.object_actuator_ids) > 0
    )
    if config.contact_guidance and not contact_guidance_enabled:
        loguru.logger.warning(
            "contact_guidance is enabled but no object actuators were resolved."
        )
    if config.warmup_steps > 0 and not warmup_enabled:
        loguru.logger.warning(
            "warmup_steps > 0 but no weld constraints found; warmup disabled."
        )
    _assert_object_actuator_gains_zero(env, config, "start")
    if warmup_enabled:
        set_weld_target(config, env)
        set_weld_active(config, env, True)
        set_gravity(env, [0.0, 0.0, 0.0])
        loguru.logger.info(
            "Warmup: enabled {} weld constraints + zero gravity for {} steps.",
            len(config.warmup_weld_eq_ids),
            config.warmup_steps,
        )
    images = []
    object_trace_site_ids = []
    robot_trace_site_ids = []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None:
            if name.startswith("trace"):
                if "object" in name:
                    object_trace_site_ids.append(sid)
                else:
                    robot_trace_site_ids.append(sid)
    config.trace_site_ids = object_trace_site_ids + robot_trace_site_ids
    config.num_object_trace_sites = len(object_trace_site_ids)
    contact_offset = 0
    if contact_guidance_enabled:
        config.contact_len = int(
            min(contact.shape[1], contact_pos.shape[1], len(config.contact_order))
        )
        if (
            config.contact_len != len(config.contact_order)
            or config.contact_len != contact.shape[1]
        ):
            loguru.logger.warning(
                "Contact length mismatch (mask={}, pos={}, expected={}); truncating to {}.",
                contact.shape[1],
                contact_pos.shape[1],
                len(config.contact_order),
                config.contact_len,
            )
        config.contact_order = config.contact_order[: config.contact_len]
        config.hand_contact_site_ids = config.hand_contact_site_ids[
            : config.contact_len
        ]
        contact_offset = max(contact.shape[1] - config.contact_len, 0)

    env_params_list = []
    if config.num_dr <= 1:
        xy_offset_list = [0.0]
        pair_margin_list = [0.0]
    else:
        xy_offset_list = np.linspace(
            config.xy_offset_range[0], config.xy_offset_range[1], config.num_dr
        )
        pair_margin_list = np.linspace(
            config.pair_margin_range[0], config.pair_margin_range[1], config.num_dr
        )
    kp_schedule = []
    kd_schedule = []
    if contact_guidance_enabled and config.max_num_iterations > 0:
        actuator_names = config.object_actuator_names
        if not actuator_names:
            actuator_names = [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
                for aid in config.object_actuator_ids
            ]
        base_kp = np.array(
            [
                (
                    config.init_rot_actuator_gain
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_gain
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        base_kd = np.array(
            [
                (
                    config.init_rot_actuator_bias
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_bias
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        for i in range(config.max_num_iterations):
            decay = float(config.guidance_decay_ratio) ** i
            kp_i = base_kp * decay
            kd_i = base_kd * decay
            if i == config.max_num_iterations - 1:
                kp_i = np.zeros_like(base_kp, dtype=np.float32)
                kd_i = np.zeros_like(base_kd, dtype=np.float32)
            kp_schedule.append(kp_i)
            kd_schedule.append(kd_i)

    for i in range(config.max_num_iterations):
        env_params = []
        for j in range(config.num_dr):
            params = {
                "xy_offset": xy_offset_list[j],
                "pair_margin": pair_margin_list[j],
            }
            if contact_guidance_enabled and kp_schedule:
                params["kp"] = kp_schedule[i]
                params["kd"] = kd_schedule[i]
            env_params.append(params)
        env_params_list.append(env_params)
    config.env_params_list = env_params_list
    _save_config_yaml(config)

    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)

    rollout = make_rollout_fn(
        step_env,
        save_state,
        load_state,
        get_reward,
        get_terminal_reward,
        get_trace,
        save_env_params,
        load_env_params,
    )
    optimize_once = make_optimize_once_fn(rollout)
    optimize = make_optimize_fn(optimize_once)
    _opt_progress_cb = None
    _viser_enabled = "viser" in (config.viewer or "")
    if _viser_enabled:
        _opt_progress_cb = _viser_viewer.update_opt_progress
        # Push the per-side execution-timeline gate state + warmup length so the
        # reward plot can shade the effective gate (green=held, red=at rest).
        # One lane per side (stacked half-height bands).
        if env._inhand_gate_mask:
            _bid_to_side = {b: s for s, b in env._side_object_body_id.items()}
            _gate_lanes = {}
            for _bid in env._inhand_gate_mask:
                _label = _bid_to_side.get(_bid, "")[:1].upper() or str(_bid)
                _gate_lanes[_label] = _gate_state_lane(
                    env, _bid, int(config.warmup_steps)
                )
            _viser_viewer.log_reward_gate(_gate_lanes, int(config.warmup_steps))
        else:
            _viser_viewer.log_reward_gate(None, int(config.warmup_steps))

    base_noise_scale = config.noise_scale.clone()
    gibbs_enabled = config.gibbs_sampling and config.embodiment_type == "bimanual"
    if config.gibbs_sampling and not gibbs_enabled:
        loguru.logger.warning(
            "gibbs_sampling is enabled but embodiment_type is {}, disabling.",
            config.embodiment_type,
        )
    if gibbs_enabled:
        right_ids, left_ids = _get_bimanual_hand_indices(config)
        right_only_zero = left_ids
        left_only_zero = right_ids

    ctrls = ctrl_ref[: config.horizon_steps]
    info_list = []

    with run_viewer() as viewer:
        # Show the initial pose before the first optimization round
        log_frame(
            mj_data,
            sim_time=0.0,
            viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
            data_ref=mj_data_ref,
            ref_color=REF_COLOR_RED if warmup_enabled else REF_COLOR_BLUE,
            record=False,
        )
        while viewer.is_running():
            # optimize using future reference window at control-rate (+1 lookahead)
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            # Tell the optimizer where we are so rollouts transition at the right step
            if warmup_enabled:
                config.warmup_sim_step = sim_step
            ref_slice = get_slice(
                ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
            )
            ctrls_for_opt = ctrls
            if contact_guidance_enabled and config.contact_len > 0:
                contact_mask_step = contact[sim_step][
                    contact_offset : contact_offset + config.contact_len
                ]
                contact_pos_ref_step = contact_pos[sim_step]
                site_xpos = wp.to_torch(env.data_wp.site_xpos)[0]

                right_delta = compute_contact_point_delta(
                    contact_mask_step,
                    contact_pos_ref_step,
                    site_xpos,
                    config.hand_contact_site_ids,
                    config.right_contact_indices,
                )
                left_delta = compute_contact_point_delta(
                    contact_mask_step,
                    contact_pos_ref_step,
                    site_xpos,
                    config.hand_contact_site_ids,
                    config.left_contact_indices,
                )
                if (
                    right_delta is not None
                    and config.right_pos_ctrl_ids
                    and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                ):
                    ctrls_for_opt = ctrls_for_opt.clone()
                    ref_ctrl_slice = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
                    ctrls_for_opt[:, config.right_pos_ctrl_ids] = ref_ctrl_slice[
                        :, config.right_pos_ctrl_ids
                    ] + torch.clip(right_delta, -0.01, 0.01)
                if (
                    left_delta is not None
                    and config.left_pos_ctrl_ids
                    and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                ):
                    if ctrls_for_opt is ctrls:
                        ctrls_for_opt = ctrls_for_opt.clone()
                        ref_ctrl_slice = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
                    ctrls_for_opt[:, config.left_pos_ctrl_ids] = ref_ctrl_slice[
                        :, config.left_pos_ctrl_ids
                    ] + torch.clip(left_delta, -0.01, 0.01)

            def _run_optimize(ctrls_in, zero_ctrl_ids):
                # zero_ctrl_ids: ctrl indices to zero in noise_scale ([]/None = no mask).
                ids = zero_ctrl_ids or []
                config.noise_scale = _apply_noise_mask(base_noise_scale, ids)
                return optimize(
                    config,
                    env,
                    ctrls_in,
                    ref_slice,
                    _opt_progress_cb,
                )

            if gibbs_enabled:
                ctrls, infos = _run_optimize(ctrls_for_opt, right_only_zero)
                ctrls, infos = _run_optimize(ctrls, left_only_zero)
            else:
                ctrls, infos = _run_optimize(ctrls_for_opt, None)
            # Restore base tensors so other consumers see the unmasked state.
            config.noise_scale = base_noise_scale

            if len(config.trace_site_ids) > 0:
                trace_ref = []
                qpos_ref_horizon = ref_slice[0]
                for h in range(config.horizon_steps):
                    mj_data_ref.qpos[:] = qpos_ref_horizon[h].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    site_xpos = np.array(
                        [mj_data_ref.site_xpos[sid] for sid in config.trace_site_ids]
                    )
                    trace_ref.append(site_xpos)

                # Make shape match trace_sample (1, 1, H, K, 3)
                # (since traces are usually I x N x P x K x 3, here I=1, N=1, P is horizon, K is sites)
                trace_ref_np = np.stack(trace_ref, axis=0)  # (H, K, 3)
                trace_ref_np = trace_ref_np[None, None, :, :, :]  # (1, 1, H, K, 3)
                infos["trace_ref"] = trace_ref_np

            step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
            _substep_max_pen = torch.zeros(env.num_worlds, device=config.device)
            for i in range(config.ctrl_steps):
                ctrl_step = ctrls[i]

                # Warmup end: disable weld and restore gravity
                abs_step = sim_step + i
                if warmup_enabled and abs_step == config.warmup_steps:
                    set_weld_active(config, env, False)
                    set_gravity(env, list(config.default_gravity))
                    loguru.logger.info(
                        "Warmup ended at step {}; weld disabled, gravity restored.",
                        abs_step,
                    )

                # Random perturbations are an optimizer-only signal for
                # finding stable grasps — disable them on outer-loop
                # execution.
                step_env(config, env, ctrl_step, perturb=False)
                # Track max penetration across all sim sub-steps (not just final snapshot)
                _pen_i, _wid_i = check_penetration(config, env)
                _pen_per_world = torch.zeros(env.num_worlds, device=config.device)
                _pen_per_world.scatter_reduce_(0, _wid_i, _pen_i, reduce="amax")
                _substep_max_pen = torch.maximum(_substep_max_pen, _pen_per_world)
                mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
                mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
                mj_data.time += config.sim_dt
                if config.save_video and renderer is not None:
                    if i % int(np.round(config.render_dt / config.sim_dt)) == 0:
                        mj_data_ref.qpos[:] = (
                            qpos_ref[sim_step + i].detach().cpu().numpy()
                        )
                        image = render_image(
                            config, renderer, mj_model, mj_data, mj_data_ref
                        )
                        images.append(image)
                if "viser" in config.viewer:
                    mujoco.mj_kinematics(mj_model, mj_data)
                    mj_data_ref.qpos[:] = qpos_ref[sim_step + i].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    frame_step = sim_step + i
                    ref_color = None
                    if warmup_enabled:
                        if frame_step < config.warmup_steps:
                            ref_color = REF_COLOR_RED
                        else:
                            ref_color = REF_COLOR_BLUE
                    log_frame(
                        mj_data,
                        sim_time=mj_data.time,
                        viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                        data_ref=mj_data_ref,
                        ref_color=ref_color,
                        playback_fps=1.0 / config.sim_dt,
                    )
                    # Per-step executed reward terms for viser plot
                    if "viser" in config.viewer:
                        _ref_i = [r[sim_step + i] for r in ref_data]
                        _rew, _info = get_reward(
                            config, env, _ref_i, sim_step=sim_step + i
                        )
                        _step_vals = {
                            k: float(_info[k][0]) for k in config._qpos_group_slices
                        }
                        _step_vals["qvel_rew"] = float(_info["qvel_rew"][0])
                        _step_vals["pen_penalty"] = float(-_info["pen_penalty"][0])
                        for _k, _v in _info.items():
                            if _k.startswith(("pedestal_", "floor_")):
                                _step_vals[_k] = float(-_v[0])
                        _viser_viewer.log_reward_step(_step_vals)
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(mj_data.ctrl.copy())
            for k in step_info:
                step_info[k] = np.stack(step_info[k], axis=0)
            infos.update(step_info)
            if warmup_enabled:
                infos["warmup_progress"] = min(
                    sim_step / max(config.warmup_steps, 1), 1.0
                )
            sync_env(config, env, mj_data)

            # receding horizon update
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            prev_ctrl = ctrls[config.ctrl_steps :]
            new_ctrl = ctrl_ref[
                sim_step + prev_ctrl.shape[0] : sim_step
                + prev_ctrl.shape[0]
                + config.ctrl_steps
            ]
            # Offset appended IK reference so controls are continuous at splice,
            # linearly decaying to zero by the end of the appended segment.
            if (
                config.blend_reference
                and new_ctrl.shape[0] > 0
                and prev_ctrl.shape[0] > 0
            ):
                delta = prev_ctrl[-1] - new_ctrl[0]
                new_ctrl = new_ctrl.clone()
                n = new_ctrl.shape[0]
                decay = torch.linspace(1.0, 0.0, n, device=new_ctrl.device).unsqueeze(
                    1
                )  # (n, 1)
                new_ctrl += delta * decay
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

            mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data_ref.qpos[:] = qpos_ref[sim_step].detach().cpu().numpy()

            infos["sim_step"] = sim_step
            update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

            info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})

            current_ref = tuple(r[sim_step] for r in ref_data)
            terminate, term_info = get_terminate(config, env, current_ref)
            # Override penetration termination with sub-step running max
            # (get_terminate only sees the final snapshot, which misses tunneling)
            terminate_pen_substep = (
                _substep_max_pen > config.terminate_penetration_threshold
            )
            terminate = terminate | terminate_pen_substep
            # Terminate if the live executed state has gone NaN/Inf —
            # planning samples that NaN are fine (masked in the optimizer),
            # but a NaN in the actual sim qpos/qvel is unrecoverable: every
            # subsequent plan reads NaN inputs and burns wall time for no
            # progress.
            _live_qpos = get_qpos(config, env)
            _live_qvel = get_qvel(config, env)
            terminate_nan = (~torch.isfinite(_live_qpos)).any(dim=-1) | (
                ~torch.isfinite(_live_qvel)
            ).any(dim=-1)
            terminate = terminate | terminate_nan
            if terminate[0]:
                loguru.logger.info(
                    "Early termination at step {}/{}: terminate condition triggered",
                    sim_step,
                    config.max_sim_steps,
                )
                break

            if sim_step >= config.max_sim_steps:
                break

    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        np.savez(
            f"{config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz",
            **info_aggregated,
        )
        loguru.logger.info(
            f"Saved info to {config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz"
        )

    if config.save_video and len(images) > 0:
        video_path = f"{config.output_dir}/visualization_mjwp{'_act' if config.contact_guidance else ''}.mp4"
        imageio.mimsave(
            video_path,
            images,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved video to {video_path}")

    errors = None
    if info_list:
        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
        qpos_ref_np = qpos_ref[: qpos_traj.shape[0]].detach().cpu().numpy()
        data_type = "mjwp_act" if config.contact_guidance else "mjwp"
        errors = compute_object_tracking_error(
            qpos_traj, qpos_ref_np, config.embodiment_type, data_type
        )
        loguru.logger.info(
            "Final object tracking error: pos={:.4f}, quat={:.4f}",
            errors["obj_pos_err"],
            errors["obj_quat_err"],
        )

    _assert_object_actuator_gains_zero(env, config, "end")

    if config.show_viewer and "viser" in config.viewer and config.wait_on_finish:
        loguru.logger.info(
            "Optimization complete! Keeping Viser server alive. Press Ctrl+C to exit."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    return errors
