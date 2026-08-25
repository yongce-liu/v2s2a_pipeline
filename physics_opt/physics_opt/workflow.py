"""Stage 5 of the retargeting pipeline: MPC physics optimization.

Ported from do-as-i-do ``launch.py``: the effective ``Config`` is YAML defaults
+ the ``do_as_i_do`` dataset override + CLI overrides, then handed to
``optimize_physics.main``. The optimizer writes ``trajectory_mjwp.npz`` and a
resolved ``config.yaml`` next to the IK inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger

from physics_opt.config import (
    Config,
    filter_config_fields,
    load_config_yaml,
)
from physics_opt.optimize_physics import main as optimize_physics

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_mjwp_config(**overrides) -> Config:
    """Build Config from YAML defaults + dataset override + caller overrides."""
    cfg_dict = load_config_yaml(str(_CONFIG_DIR / "default.yaml"))
    override_path = _CONFIG_DIR / "override" / "do_as_i_do.yaml"
    if override_path.exists():
        cfg_dict.update(load_config_yaml(str(override_path)))
    cfg_dict.update(overrides)

    filtered = filter_config_fields(cfg_dict)
    for key in ("pair_margin_range", "xy_offset_range"):
        if key in filtered:
            filtered[key] = tuple(filtered[key])
    filtered.pop("noise_scale", None)
    return Config(**filtered)


@dataclass
class PhysicsOverrideArgs:
    """Hand-picked Config overrides; None keeps the YAML value."""

    num_samples: int | None = None
    max_num_iterations: int | None = None
    sim_dt: float | None = None
    ctrl_dt: float | None = None
    horizon: float | None = None


@dataclass
class PhysicsOptArgs:
    """Run sampling-based MPC physics optimization (MuJoCo Warp) for a task."""

    task: str = ""
    """Video/task name (as resolved by scene_construction stage 1)."""

    output_root: Path = (
        Path(__file__).parents[2] / "outputs" / "yellow_spoon" / "scene_construction"
    )
    """Retargeting output root containing stages 1-4.5 artifacts; must match
    the scene_construction/retarget runs (default
    ``outputs/<clip>/scene_construction``)."""

    hand_type: Literal["auto", "left", "right", "bimanual"] = "auto"
    """Hand embodiment (must match the retarget run)."""

    robot_type: str = "sharpa"
    """Target robot hand (must match the retarget run)."""

    data_id: int = 0
    """Trial index under the task directory."""

    dataset_name: str = "do_as_i_do"
    """Dataset tag (must match earlier stages)."""

    seed: int = 0
    """Optimizer random seed."""

    max_sim_steps: int = 0
    """Bound the optimization length (0 = full trajectory)."""

    show_viewer: bool = True
    """Serve the viser viewer during optimization."""

    wait_on_finish: bool = True
    """Keep the viewer alive after the run (Ctrl+C to exit)."""

    save_video: bool = False
    """Render the optimized and reference trajectories to an MP4."""

    output_subdir: str = "physics_opt"
    """Subdirectory under the trial dir for stage-5 artifacts
    (``trajectory_mjwp.npz`` etc.); empty writes beside the IK inputs."""

    force: bool = True
    """Overwrite an existing trajectory_mjwp.npz."""

    override: PhysicsOverrideArgs = field(default_factory=PhysicsOverrideArgs)
    """Additional Config field overrides (applied after the YAML layers)."""


def run_physics_opt(args: PhysicsOptArgs) -> None:
    """Build the Config and run the MPC optimization."""
    override_dict = {k: v for k, v in vars(args.override).items() if v is not None}
    config = load_mjwp_config(
        dataset_name=args.dataset_name,
        task=args.task,
        data_id=args.data_id,
        robot_type=args.robot_type,
        embodiment_type=args.hand_type,
        output_root_dir=str(args.output_root),
        seed=args.seed,
        wait_on_finish=args.wait_on_finish,
        max_sim_steps=args.max_sim_steps,
        force=args.force,
        show_viewer=args.show_viewer,
        save_video=args.save_video,
        output_subdir=args.output_subdir,
        **override_dict,
    )
    optimize_physics(config)
    logger.info("[physics_opt] optimization complete: task={}", args.task)
