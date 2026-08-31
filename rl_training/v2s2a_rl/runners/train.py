"""Train with the IsaacLab 3.0 public launcher and RSL-RL 5 runner."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import v2s2a_rl.tasks  # noqa: F401
from v2s2a_rl.runners import cli_args

parser = argparse.ArgumentParser(description="Train a v2s2a trajectory policy with RSL-RL PPO")
parser.add_argument("--task", required=True)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument(
    "--video_interval",
    type=int,
    default=None,
    help="Record every N environment/control steps.",
)
parser.add_argument(
    "--video_every_iterations",
    type=int,
    default=50,
    help="Record every N PPO iterations (converted using num_steps_per_env).",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        env_cfg.seed = agent_cfg.seed

    with launch_simulation(env_cfg, args_cli):
        video_interval = args_cli.video_interval
        if video_interval is None:
            video_interval = args_cli.video_every_iterations * agent_cfg.num_steps_per_env
        if video_interval <= 0:
            raise ValueError("video interval must be positive")
        run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            run += f"_{agent_cfg.run_name}"
        log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, run))
        env_cfg.log_dir = log_dir
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(log_dir, "videos", "train"),
                step_trigger=lambda step: step % video_interval == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        if agent_cfg.resume:
            checkpoint = args_cli.checkpoint or agent_cfg.load_checkpoint
            if not checkpoint:
                raise ValueError("--resume requires --checkpoint")
            runner.load(os.path.abspath(checkpoint))
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        # Task phase is already randomized explicitly in the environment; random
        # Gym episode lengths would desynchronize phase and timeout semantics.
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
        env.close()
        print(f"[INFO] Checkpoints: {log_dir}")


if __name__ == "__main__":
    main()
