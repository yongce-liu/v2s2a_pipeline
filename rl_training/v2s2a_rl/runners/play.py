"""Evaluate, record and export a trained v2s2a policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import v2s2a_rl.tasks  # noqa: F401
from v2s2a_rl.runners import cli_args

parser = argparse.ArgumentParser(description="Evaluate a v2s2a RSL-RL policy")
parser.add_argument("--task", required=True)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--zero-policy", action="store_true", help="evaluate the reference controller baseline")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--real-time", action="store_true")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    log_dir = checkpoint.parent

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(log_dir / "videos" / "eval"),
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if args_cli.zero_policy:
            class ZeroPolicy:
                def __call__(self, obs):
                    first = next(iter(obs.values())) if isinstance(obs, dict) else obs
                    action_dim = gym.spaces.flatdim(env.unwrapped.single_action_space)
                    return torch.zeros(first.shape[0], action_dim, device=first.device)

                def reset(self, dones):
                    return None

            policy = ZeroPolicy()

        export_dir = log_dir / "exported"
        runner.export_policy_to_jit(path=str(export_dir), filename="policy.pt")
        runner.export_policy_to_onnx(path=str(export_dir), filename="policy.onnx")

        obs = env.get_observations()
        completed = 0
        successes = 0
        total_reward = 0.0
        steps = 0
        rollout_steps = 0
        while completed < args_cli.episodes:
            start = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, reward, dones, extras = env.step(actions)
                policy.reset(dones)
            total_reward += float(reward.sum())
            steps += int(reward.numel())
            rollout_steps += 1
            if torch.any(dones):
                done_count = int(dones.sum())
                completed += done_count
                terminal_success = extras.get("terminal_success")
                if terminal_success is not None:
                    successes += int(terminal_success.sum())
                else:
                    metric = extras.get("log", {}).get("Metrics/episode_success_rate", 0.0)
                    successes += round(float(metric) * done_count)
            if args_cli.video and rollout_steps >= args_cli.video_length:
                break
            delay = env.unwrapped.step_dt - (time.time() - start)
            if args_cli.real_time and delay > 0:
                time.sleep(delay)

        report = {
            "checkpoint": str(checkpoint),
            "episodes": completed,
            "successes": successes,
            "success_rate": successes / max(1, completed),
            "mean_step_reward": total_reward / max(1, steps),
            "policy_jit": str(export_dir / "policy.pt"),
            "policy_onnx": str(export_dir / "policy.onnx"),
        }
        report_path = log_dir / "evaluation.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        env.close()


if __name__ == "__main__":
    main()
