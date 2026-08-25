"""Command-line entry point for the process package.

Usage: ``uv run python -m process.cli --video-path inputs/a.mp4``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from loguru import logger

from process.extract import ExtractArgs
from process.workflow import run_ingest


@dataclass
class ProcessCliArgs:
    """Ingest one source video: probe its format and extract frames."""

    video_path: Path
    """Source video to ingest (treated as read-only)."""

    output_root: Path = Path(__file__).parents[2] / "outputs"
    """Root under which ``<video_stem>/process/`` is created."""

    extract: ExtractArgs = field(default_factory=ExtractArgs)
    """Frame extraction settings."""


def main() -> None:
    args = tyro.cli(ProcessCliArgs)
    run_ingest(
        video_path=args.video_path,
        output_root=args.output_root,
        extract_args=args.extract,
    )
    logger.info("[process] ingest complete: {}", args.video_path)


if __name__ == "__main__":
    main()
