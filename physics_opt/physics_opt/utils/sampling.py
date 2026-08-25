"""Noise schedule + DIAL-MPC sampling/rollout for the optimizer."""

from __future__ import annotations

from collections.abc import Callable

import loguru
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from physics_opt.config import Config
from physics_opt.utils.interp import interp


def _stage_joint_delta(
    config,
    global_noise_scale: torch.Tensor,
) -> torch.Tensor:
    """Per-actuator joint-space exploration delta, (num_samples, horizon, nu)."""
    knot_samples = (
        torch.randn_like(config.noise_scale, device=config.device)
        * config.noise_scale
        * global_noise_scale
    )
    return interp(knot_samples, config.knot_steps)


def _sample_ctrls_impl(
    config,
    ctrls: torch.Tensor,
    global_noise_scale: torch.Tensor,
) -> torch.Tensor:
    return ctrls + _stage_joint_delta(config, global_noise_scale)


# Compiled variant (torch.compile requires PyTorch 2.0+).
if hasattr(torch, "compile"):
    _sample_ctrls_compiled = torch.compile(_sample_ctrls_impl)
else:
    _sample_ctrls_compiled = _sample_ctrls_impl


def sample_ctrls(
    config, ctrls: torch.Tensor, sample_params: dict | None = None
) -> torch.Tensor:
    """Sample (num_samples, horizon, nu) ctrls from (horizon, nu) reference."""
    gns_value = sample_params.get("global_noise_scale", 1.0) if sample_params else 1.0
    if torch.is_tensor(gns_value):
        gns = gns_value.to(device=config.device)
    else:
        gns = torch.tensor(float(gns_value), device=config.device)
    if config.use_torch_compile:
        return _sample_ctrls_compiled(config, ctrls, gns)
    return _sample_ctrls_impl(config, ctrls, gns)


def make_rollout_fn(
    step_env,
    save_state,
    load_state,
    get_reward,
    get_terminal_reward,
    get_trace,
    save_env_params,
    load_env_params,
):
    def rollout(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_param: dict,
    ) -> torch.Tensor:
        """Roll out the control actions; returns (ctrls, reward (N,), info)."""
        init_state = save_state(env)
        init_env_param = save_env_params(config, env)
        env = load_env_params(config, env, env_param)

        # Each rollout is an independent evaluation of candidate controls from
        # the same starting state — clear any perturbation state left over from
        # the previous rollout so candidates are not "pre-disturbed" by
        # unrelated hypothetical histories.
        for _state in (env._perturb_active, env._perturb_force, env._perturb_torque):
            for _tensor in _state.values():
                _tensor.zero_()

        # N control candidates run as N*K parallel worlds (K = num_perturb_samples),
        # so each candidate is evaluated under K disturbance realizations and averaged.
        N, H = ctrls.shape[:2]
        K = int(config.num_perturb_samples)
        # Expand controls so world n*K+k runs candidate n (matches the
        # n*K+k → seed-k random-draw layout in apply_perturbation).
        ctrls_world = ctrls.repeat_interleave(K, dim=0) if K > 1 else ctrls
        trace_list = []
        cum_rew = torch.zeros(N * K, device=config.device)
        cum_info: dict[str, torch.Tensor] | None = None
        _warmup_transition_t = -1
        _warmup_rollout_steps = 0  # rollout steps inside warmup (reward down-weighted)
        _warmup_needs_cleanup = False
        if config.warmup_steps > 0 and config.warmup_sim_step <= config.warmup_steps:
            from physics_opt.utils.mjwp import set_gravity as _set_gravity
            from physics_opt.utils.mjwp import set_weld_active as _set_weld_active

            if config.warmup_sim_step < config.warmup_steps:
                # Rollout step t where warmup ends (absolute step = sim_step + t + 1)
                _warmup_transition_t = config.warmup_steps - config.warmup_sim_step - 1
                if _warmup_transition_t >= H:
                    _warmup_rollout_steps = H  # entire rollout is in warmup
                    _warmup_transition_t = -1  # transition is beyond this rollout
                else:
                    _warmup_rollout_steps = max(_warmup_transition_t, 0)
                # Re-enable weld: a previous rollout iteration may have disabled it
                # it at the transition step, and load_state does not restore
                # eq_active (model-level state).
                _set_weld_active(config, env, True)
                _warmup_needs_cleanup = True
            else:
                # sim_step == warmup_steps: warmup just ended but the weld may
                # still be active (the outer loop disables it during stepping,
                # which runs AFTER the optimizer).  Disable it now so the
                # rollout runs clean post-warmup physics.
                _set_weld_active(config, env, False)
                _set_gravity(env, list(config.default_gravity))
                _warmup_needs_cleanup = True

        for t in range(H):
            abs_step = config.warmup_sim_step + t + 1

            # Warmup end: disable weld and ensure full gravity
            if t == _warmup_transition_t:
                _set_weld_active(config, env, False)
                _set_gravity(env, list(config.default_gravity))
            step_env(
                config,
                env,
                ctrls_world[:, t],
                perturb=(
                    abs_step >= config.warmup_steps
                ),  # no perturbation during warmup
                sim_step=abs_step,
            )
            ref = [r[t] for r in ref_slice]
            rew, step_info = (
                get_reward(config, env, ref, sim_step=abs_step)
                if t < H - 1
                else get_terminal_reward(config, env, ref, sim_step=abs_step)
            )
            if t < _warmup_rollout_steps:
                cum_rew += config.warmup_rew_multiplier * rew
            else:
                cum_rew += rew
            # running sums avoid H dict allocations + a stack
            if cum_info is None:
                cum_info = {k: v.clone() for k, v in step_info.items()}
            else:
                for k in cum_info:
                    cum_info[k] += step_info[k]
            trace = get_trace(config, env)
            trace_list.append(trace)

        # Reduce over the K perturbation seeds: each candidate's reward and
        # info entries are averaged across its K replicas to give the
        # variance-reduced per-candidate estimate.
        effective_H = (
            H - _warmup_rollout_steps
        ) + config.warmup_rew_multiplier * _warmup_rollout_steps
        mean_rew = (cum_rew / max(effective_H, 1)).view(N, K).mean(dim=-1)
        mean_info = (
            {k: (v / H).view(N, K).mean(dim=-1) for k, v in cum_info.items()}
            if cum_info
            else {}
        )

        # Restore weld and gravity (model-level, not covered by load_state)
        if _warmup_needs_cleanup:
            if config.warmup_sim_step < config.warmup_steps:
                _set_weld_active(config, env, True)
            _set_gravity(env, list(config.default_gravity))

        env = load_state(env, init_state)
        env = load_env_params(config, env, init_env_param)

        # trace_list stacks to (N*K, H, n_trace, 3); take the k=0 replica per
        # candidate for visualization (averaging path points across seeds
        # would produce non-physical interpolated trajectories).
        trace_stacked = torch.stack(trace_list, dim=1)
        trace_repr = (
            trace_stacked.view(N, K, *trace_stacked.shape[1:])[:, 0]
            if K > 1
            else trace_stacked
        )
        info = {
            "trace": trace_repr,  # (N, H, n_trace, 3)
            **mean_info,
        }
        return ctrls, mean_rew, info

    return rollout


def _compute_weights_impl(
    rews: torch.Tensor, num_samples: int, temperature: float
) -> torch.Tensor:
    """Softmax weights from rewards; returns (weights (N,), nan_mask)."""
    # Replace NaN/inf rewards with the min finite reward (or -1000 if all bad).
    nan_mask = torch.isnan(rews) | torch.isinf(rews)
    safe_rews = torch.where(nan_mask, torch.tensor(torch.inf, device=rews.device), rews)
    rews_min = safe_rews.min()
    rews_min = torch.where(
        torch.isinf(rews_min), torch.tensor(-1000.0, device=rews.device), rews_min
    )
    rews = torch.where(nan_mask, rews_min, rews)

    # Softmax over only the top 10% of samples.
    top_k = max(1, int(0.1 * num_samples))
    top_indices = torch.topk(rews, k=top_k, largest=True).indices

    weights = torch.zeros_like(rews)
    top_rews = rews[top_indices]
    top_rews_normalized = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
    top_weights = F.softmax(top_rews_normalized / temperature, dim=0)
    weights[top_indices] = top_weights

    return weights, nan_mask


# Compiled version (torch.compile requires PyTorch 2.0+)
if hasattr(torch, "compile"):
    _compute_weights_compiled = torch.compile(_compute_weights_impl)
else:
    _compute_weights_compiled = _compute_weights_impl


def make_optimize_once_fn(
    rollout,
):
    def optimize_once(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_params: list[dict] = [{}],
        sample_params: dict | None = None,
    ) -> torch.Tensor:
        """One DIAL-MPC optimization step (no annealing)."""
        ctrls_samples = sample_ctrls(config, ctrls, sample_params)

        # Domain randomization: worst-case (min) reward across all DR parameter sets.
        min_rew = torch.full((config.num_samples,), float("inf"), device=config.device)
        for env_param in env_params:
            ctrls_samples, rews, rollout_info = rollout(
                config,
                env,
                ctrls_samples,
                ref_slice,
                env_param,
            )
            min_rew = torch.minimum(min_rew, rews)
        rews = min_rew

        if config.use_torch_compile:
            weights, nan_mask = _compute_weights_compiled(
                rews, config.num_samples, config.temperature
            )
        else:
            weights, nan_mask = _compute_weights_impl(
                rews, config.num_samples, config.temperature
            )
        if nan_mask.any():
            loguru.logger.warning(
                f"NaNs or infs in rews: {nan_mask.sum()}/{config.num_samples}"
            )

        ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)

        # Downsample traces (topk + uniform samples) for visualization.
        n_uni = max(0, min(config.num_trace_uniform_samples, config.num_samples))
        n_topk = max(0, min(config.num_trace_topk_samples, config.num_samples))
        idx_uni = (
            torch.linspace(
                0,
                config.num_samples - 1,
                steps=n_uni,
                dtype=torch.long,
                device=config.device,
            )
            if n_uni > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        idx_top = (
            torch.topk(rews, k=n_topk, largest=True).indices
            if n_topk > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        sel_idx = torch.cat([idx_uni, idx_top], dim=0).long()

        info = {}
        for k, v in rollout_info.items():
            if k not in ["trace", "trace_sample"]:
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                if v.ndim == 1:
                    k_max = k + "_max"
                    k_min = k + "_min"
                    k_median = k + "_median"
                    k_mean = k + "_mean"
                    info[k_max] = v.max()
                    info[k_min] = v.min()
                    info[k_median] = np.median(v)
                    info[k_mean] = v.mean()
        rews_np = rews.cpu().numpy()
        info["improvement"] = rews_np.max() - rews_np[0]
        info["rew_max"] = rews_np.max()
        info["rew_min"] = rews_np.min()
        info["rew_median"] = np.median(rews_np)
        info["rew_mean"] = rews_np.mean()

        if "trace" in rollout_info:
            info["trace_sample"] = (
                rollout_info["trace"][sel_idx].cpu().numpy()
            )  # (M, H, n_trace, 3)
            info["trace_cost"] = -rews[sel_idx].cpu().numpy()

        return ctrls_mean, info

    return optimize_once


def make_optimize_fn(
    optimize_once,
):
    def optimize(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        """Full optimization loop at certain time step."""
        infos = []

        sample_params_list = [
            {"global_noise_scale": config.beta_traj**i}
            for i in range(config.max_num_iterations)
        ]

        improvement_history = []
        pbar = tqdm(range(config.max_num_iterations), desc="Optimizing", leave=False)
        if progress_callback is not None:
            progress_callback(0, config.max_num_iterations)
        for i in pbar:
            ctrls, info = optimize_once(
                config,
                env,
                ctrls,
                ref_slice,
                config.env_params_list[i],
                sample_params_list[i],
            )
            infos.append(info)
            improvement_history.append(info["improvement"])
            pbar.set_postfix(
                rew=f"{info['rew_max']:.4f}" if "rew_max" in info else "N/A",
                imp=f"{info['improvement']:.4f}",
            )
            if progress_callback is not None:
                progress_callback(i + 1, config.max_num_iterations)

            # Early stop once the last n steps all improve below threshold.
            if len(improvement_history) >= config.improvement_check_steps:
                recent_improvements = improvement_history[
                    -config.improvement_check_steps :
                ]
                if all(
                    imp < config.improvement_threshold for imp in recent_improvements
                ):
                    break

        # Pad infos with zeros up to max_num_iterations (early stop shortens it).
        fake_info = {}
        for k, v in infos[0].items():
            fake_info[k] = np.zeros_like(v)
        for _ in range(config.max_num_iterations - len(infos)):
            infos.append(fake_info)
        info_aggregated = {}
        for k in infos[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in infos], axis=0)
        info_aggregated["opt_steps"] = np.array([i + 1])
        return ctrls, info_aggregated

    return optimize
