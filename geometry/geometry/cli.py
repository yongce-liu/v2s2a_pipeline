"""Command-line entry point for the geometry package.

Two modes, dispatched with ``--command``:

- ``single``: MoGe geometry for one image.
- ``video``:  frame-by-frame geometry for a whole video, reading the
  ``process`` stage's ``frames.json``.

Usage:

.. code-block:: bash

    # Single image
    uv run python -m geometry.cli --command single \
        --single.image-path frame.png --single.output-dir out

    # Full video (frame-by-frame), reading process frames.json
    uv run python -m geometry.cli --command video \
        --video.frames-json outputs/0/process/frames.json \
        --video.vis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from geometry.moge_model import MogeArgs
from geometry.workflow import (
    GeometryVideoArgs,
    process_single_image,
    run_video_geometry,
)


@dataclass
class SingleImageArgs:
    """Inputs for ``--command single`` (validated only when that mode runs)."""

    image_path: Path | None = None
    output_dir: Path | None = None
    moge: MogeArgs = field(default_factory=MogeArgs)


@dataclass
class GeometryCliArgs:
    """Run MoGe monocular geometry estimation, single image or full video."""

    command: Literal["single", "video"] = "video"
    """Geometry mode: one image (``single``) or a whole video (``video``)."""

    single: SingleImageArgs = field(default_factory=SingleImageArgs)
    """Settings for ``--command single``."""

    video: GeometryVideoArgs = field(default_factory=GeometryVideoArgs)
    """Settings for ``--command video``."""


def main() -> None:
    args = tyro.cli(GeometryCliArgs)

    if args.command == "single":
        if args.single.image_path is None or args.single.output_dir is None:
            raise ValueError(
                "--command single requires --single.image-path and --single.output-dir."
            )
        outputs = process_single_image(
            image_path=args.single.image_path,
            output_dir=args.single.output_dir,
            args=args.single.moge,
        )
        logger.info(
            "[geometry] single-image geometry complete: out={}",
            outputs.output_dir,
        )
        return

    run_video_geometry(args.video)
    logger.info("[geometry] video geometry complete: {}", args.video.frames_json)


if __name__ == "__main__":
    main()
