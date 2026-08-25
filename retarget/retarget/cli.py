"""Command-line entry point for the retarget package.

Usage: ``uv run retarget --task whisking`` (after scene_construction has run).
"""

from __future__ import annotations

import tyro

from retarget.workflow import RetargetArgs, run_retarget


def main() -> None:
    args = tyro.cli(RetargetArgs)
    run_retarget(args)


if __name__ == "__main__":
    main()
