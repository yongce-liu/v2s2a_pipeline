"""RSL-RL CLI overrides kept compatible with IsaacLab 3.0 runners."""

from __future__ import annotations

import argparse
import random


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rsl_rl")
    group.add_argument("--experiment_name", default=None)
    group.add_argument("--run_name", default=None)
    group.add_argument("--resume", action="store_true", default=False)
    group.add_argument("--load_run", default=None)
    if "--checkpoint" not in parser._option_string_actions:
        group.add_argument("--checkpoint", default=None)
    group.add_argument("--logger", choices={"wandb", "tensorboard", "neptune"}, default=None)
    group.add_argument("--log_project_name", default=None)


def update_rsl_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    if getattr(args_cli, "seed", None) is not None:
        agent_cfg.seed = random.randint(0, 10_000) if args_cli.seed == -1 else args_cli.seed
    agent_cfg.resume = bool(getattr(args_cli, "resume", False))
    if getattr(args_cli, "load_run", None) is not None:
        agent_cfg.load_run = args_cli.load_run
    if getattr(args_cli, "checkpoint", None) is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if getattr(args_cli, "run_name", None) is not None:
        agent_cfg.run_name = args_cli.run_name
    if getattr(args_cli, "logger", None) is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and getattr(args_cli, "log_project_name", None):
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if getattr(args_cli, "experiment_name", None) is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    return agent_cfg
