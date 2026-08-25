"""Retargeting stages 4-4.5: mink IK, then pedestal/support resolution.

Stage 4 solves the hand trajectory against the pedestal-free structural scene
(``scene_ik.xml``) with mink; stage 4.5 injects stabilizing pedestals using the
IK-output object pose and emits ``scene.xml`` (+ ``scene_eq.xml``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

from retarget.resolve_pedestal import resolve_scene_pedestal
from retarget.solve_ik import main as solve_ik


@dataclass
class RetargetArgs:
    """Inputs for stage 4 (IK) and 4.5 (pedestal resolution)."""

    task: str = ""
    """Video/task name (as resolved by scene_construction stage 1)."""

    output_root: Path = (
        Path(__file__).parents[2] / "outputs" / "yellow_spoon" / "scene_construction"
    )
    """Retargeting output root containing the stage 1-3 artifacts; must match
    the scene_construction run's ``--output-root`` (default
    ``outputs/<clip>/scene_construction``)."""

    hand_type: Literal["auto", "left", "right", "bimanual"] = "auto"
    """Hand embodiment (must match the scene_construction run)."""

    robot_type: str = "sharpa"
    """Target robot hand (must match the scene_construction run)."""

    data_id: int = 0
    """Trial index under the task directory."""

    dataset_name: str = "do_as_i_do"
    """Dataset tag (must match the scene_construction run)."""

    smoothing: bool = True
    """Box-filter the IK qpos trajectory before saving."""

    save_video: bool = False
    """Render the IK result to visualization_ik.mp4."""

    hand_object_distance_thresh: float = 0.1
    """In-hand threshold (m) gating pedestal placement; mirror it into the
    physics_opt stage via its ``hand_object_distance_thresh`` config."""

    run_pedestal: bool = True
    """Also run stage 4.5 (scene_ik.xml -> scene.xml + scene_eq.xml)."""

    force: bool = True
    """Overwrite existing stage outputs."""


def run_retarget(args: RetargetArgs) -> None:
    """Run stage 4 (mink IK) and optionally stage 4.5 (pedestals)."""
    solve_ik(
        task=args.task,
        dataset_name=args.dataset_name,
        data_id=args.data_id,
        output_root_dir=str(args.output_root),
        embodiment_type=args.hand_type,
        robot_type=args.robot_type,
        show_viewer=False,
        save_video=args.save_video,
        force=args.force,
        smoothing=args.smoothing,
    )

    if args.run_pedestal:
        resolve_scene_pedestal(
            output_root_dir=str(args.output_root),
            dataset_name=args.dataset_name,
            robot_type=args.robot_type,
            embodiment_type=args.hand_type,
            task=args.task,
            data_id=args.data_id,
            use_pedestal=True,
            use_support=True,
            hand_object_distance_thresh=args.hand_object_distance_thresh,
            force=args.force,
        )
    logger.info("[retarget] IK complete: task={}, robot={}", args.task, args.robot_type)
