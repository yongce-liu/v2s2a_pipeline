"""Command-line entry point for hand-object alignment."""

from __future__ import annotations

import tyro
from loguru import logger

from hand_object_alignment.workflow import AlignmentArgs, run_alignment


def main() -> None:
    args = tyro.cli(AlignmentArgs)
    outputs = run_alignment(args)
    logger.info("[hand_object_alignment] wrote {}", outputs.poses_json_path)


if __name__ == "__main__":
    main()
