"""Command-line entry point for the pose_estimation package.

Usage:

.. code-block:: bash

    uv run python -m pose_estimation.cli \
        --frames-json outputs/0/process/frames.json \
        --mesh-path outputs/0/obj_recon/yellow_spoon/mesh.obj \
        --masks-json outputs/0/segment/masks.json
"""

from __future__ import annotations

import tyro
from loguru import logger

from pose_estimation.workflow import PoseEstimationVideoArgs, run_video_pose_estimation


def main() -> None:
    args = tyro.cli(PoseEstimationVideoArgs)
    run_video_pose_estimation(args)
    logger.info("[pose_estimation] complete: {}", args.frames_json)


if __name__ == "__main__":
    main()
