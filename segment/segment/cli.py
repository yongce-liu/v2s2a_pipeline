"""Command-line entry point for the segment package.

Two modes, dispatched with ``--command``:

- ``single``: SAM3 hand-mask segmentation of one image (existing behavior).
- ``video``:  frame-by-frame segmentation of a whole video, reading the
  ``process`` stage's ``frames.json``.

Usage:

.. code-block:: bash

    # Single image
    uv run python -m segment.cli --command single \
        --single.image-path frame.png --single.output-dir out --single.sam-mask.checkpoint ckpt.pt

    # Full video (frame-by-frame), reading process frames.json
    uv run python -m segment.cli --command video \
        --video.frames-json outputs/0/process/frames.json \
        --video.vis --video.sam-mask.checkpoint ckpts/sam3/sam3.pt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger

from segment.sam_mask import SamMaskArgs, process_sam_mask
from segment.workflow import SegmentVideoArgs, run_video_segment


@dataclass
class SingleImageArgs:
    """Inputs for ``--command single`` (validated only when that mode runs)."""

    image_path: Path | None = None
    output_dir: Path | None = None
    sam_mask: SamMaskArgs = field(default_factory=SamMaskArgs)


@dataclass
class SegmentCliArgs:
    """Run SAM3 hand-mask segmentation, single image or full video."""

    command: Literal["single", "video"] = "video"
    """Segmentation mode: one image (``single``) or a whole video (``video``)."""

    single: SingleImageArgs = field(default_factory=SingleImageArgs)
    """Settings for ``--command single``."""

    video: SegmentVideoArgs = field(default_factory=SegmentVideoArgs)
    """Settings for ``--command video``."""


def main() -> None:
    args = tyro.cli(SegmentCliArgs)

    if args.command == "single":
        if args.single.image_path is None or args.single.output_dir is None:
            raise ValueError(
                "--command single requires --single.image-path and --single.output-dir."
            )
        outputs = process_sam_mask(
            image_path=args.single.image_path,
            output_dir=args.single.output_dir,
            args=args.single.sam_mask,
        )
        logger.info(
            "[segment] single-image mask complete: mask={}, overlay={}",
            outputs.mask_path,
            outputs.overlay_path,
        )
        return

    run_video_segment(args.video)
    logger.info("[segment] video segmentation complete: {}", args.video.frames_json)


if __name__ == "__main__":
    main()
