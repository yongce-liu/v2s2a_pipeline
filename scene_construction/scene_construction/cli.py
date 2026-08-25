"""Command-line entry point for the scene_construction package.

Usage: ``uv run scene_construction --task whisking --raw-dir ../reconstruction/whisking``
"""

from __future__ import annotations

import tyro
from loguru import logger

from scene_construction.workflow import SceneBuildArgs, run_scene_build


def main() -> None:
    args = tyro.cli(SceneBuildArgs)
    run_scene_build(args)
    logger.info("[scene_construction] done: {}", args.task)


if __name__ == "__main__":
    main()
