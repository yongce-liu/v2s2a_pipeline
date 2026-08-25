"""Command-line entry point for the physics_opt package.

Usage: ``uv run physics_opt --task whisking`` (after retarget has run).
"""

from __future__ import annotations

import tyro

from physics_opt.workflow import PhysicsOptArgs, run_physics_opt


def main() -> None:
    args = tyro.cli(PhysicsOptArgs)
    run_physics_opt(args)


if __name__ == "__main__":
    main()
