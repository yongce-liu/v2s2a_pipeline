"""Command-line entry point for the hand_recon package.

Usage:

.. code-block:: bash

    uv run python -m hand_recon.cli \
        --frames-json outputs/0/process/frames.json
"""

from __future__ import annotations

import tyro
from loguru import logger

from hand_recon.workflow import HandReconArgs, run_hand_recon


def main() -> None:
    args = tyro.cli(HandReconArgs)
    outputs = run_hand_recon(args)
    logger.info(
        "[hand_recon] stage complete: meshes={} hands={} overlay={}",
        outputs.meshes_npz_path,
        outputs.hands_json_path,
        outputs.vis_overlay_mp4,
    )


if __name__ == "__main__":
    main()
