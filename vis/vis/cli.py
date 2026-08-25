"""Command-line entry point for the vis package — browser replay of pipeline outputs.

Two modes, selected with ``--mode``:

- ``scene``: MuJoCo scene replay (``scene.xml`` + physics/IK trajectory ``.npz``)
  with MANO-hand / object-reference overlays, in the scene world frame.
- ``raw``: camera-frame reconstruction debug — deforming MANO hands, tracked
  object mesh, geometry point cloud + camera frustum — straight from the
  hand_recon / pose_estimation / geometry stage outputs.

Every input is discovered from ``outputs/<clip>/`` and can be pinned with an
explicit override flag. Run from the repo root after ``cd vis && uv sync``:

.. code-block:: bash

    vis/.venv/bin/python -m vis.cli --clip-root outputs/yellow_spoon --mode raw
    vis/.venv/bin/python -m vis.cli --clip-root outputs/yellow_spoon --mode scene

Then open http://localhost:<port> and use the Frame slider / Play button.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from loguru import logger


@dataclass
class SceneOverrides:
    """Optional path overrides for ``--mode scene`` (run-dir discovery inputs)."""

    run_dir: Path | None = None
    """Run dir containing scene.xml + trajectory .npz; skips robot/data_id discovery."""

    scene: Path | None = None
    """Override path to scene.xml."""

    traj: Path | None = None
    """Override path to the trajectory .npz (mjwp or kinematic)."""

    ik: Path | None = None
    """Override path to trajectory_kinematic.npz (IK ghost layer)."""

    mano: Path | None = None
    """Override path to trajectory_keypoints.npz (MANO/object reference layer)."""

    mesh: Path | None = None
    """Override path to the object visual mesh (metric)."""

    embodiment: str | None = None
    """Override embodiment (default: discovered from scene_construction/mano/*/<task>)."""

    robot: str | None = None
    """Override robot name (default: discovered from the scene.xml location)."""

    data_id: str = "0"
    """Data id directory name under <task>/."""


@dataclass
class RawOverrides:
    """Optional path overrides for ``--mode raw`` (camera-frame debug inputs)."""

    meshes_npz: Path | None = None
    """Override path to hand_recon meshes.npz (deforming MANO hands)."""

    poses_json: Path | None = None
    """Override path to pose_estimation poses.json (object ob_in_cam trajectory)."""

    geometry_json: Path | None = None
    """Override path to geometry geometry.json (point clouds + intrinsics)."""

    mesh: Path | None = None
    """Override path to the object mesh (obj_recon canonical units)."""

    hands_json: Path | None = None
    """Override path to hand_recon hands.json (camera intrinsics)."""

    max_points: int = 50_000
    """Maximum point-cloud points kept per frame (random subsample beyond this)."""


@dataclass
class VisArgs:
    """Replay finished pipeline runs in the browser (viser)."""

    clip_root: Path = Path("outputs/yellow_spoon")
    """Clip output root (``outputs/<clip>``) everything is discovered from."""

    mode: Literal["scene", "raw"] = "scene"
    """``scene``: MuJoCo replay + world-frame layers. ``raw``: camera-frame debug."""

    port: int = 8081
    """Viser server port."""

    fps: float = 60.0
    """Initial playback FPS (frames per second on the displayed timeline)."""

    skip_warmup: bool = True
    """Drop the leading warmup frames from config.yaml (scene mode)."""

    scene: SceneOverrides = field(default_factory=SceneOverrides)
    """Path overrides for ``--mode scene``."""

    raw: RawOverrides = field(default_factory=RawOverrides)
    """Path overrides for ``--mode raw``."""


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = tyro.cli(VisArgs)
    clip_root = args.clip_root.resolve()
    if not clip_root.is_dir():
        raise SystemExit(f"Clip root not found: {clip_root}")

    # Deferred so `--help` and tests don't need viser/mujoco installed.
    from vis.workflow import run_raw, run_scene

    logger.info("[vis] clip={} mode={} port={}", clip_root, args.mode, args.port)
    if args.mode == "scene":
        run_scene(args, clip_root)
    else:
        run_raw(args, clip_root)


if __name__ == "__main__":
    main()
