"""End-to-end scene-construction stages (do-as-i-do retargeting stages 1-3).

Stage 1 builds qpos trajectories from the v2s2a stage outputs under a clip
root (``outputs/<clip>/{hand_recon,pose_estimation,...}``); stage 2
convex-decomposes the object mesh; stage 3 assembles the MuJoCo scene (robot
hand + object + contact pairs, plus the UR3 wrist-housing cylinders). The
output layout under ``output_root`` is identical to do-as-i-do's ``outputs/``
so downstream stages are interchangeable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from scene_construction.gravity import estimate_gravity_from_frames
from scene_construction.motion_sources import ClipInputs, load_clip_inputs
from scene_construction.pipeline.decompose_mesh import main as decompose_mesh
from scene_construction.pipeline.generate_scene import main as generate_scene
from scene_construction.pipeline.process_dataset import main as process_dataset


@dataclass
class SceneBuildArgs:
    """Inputs for stages 1-3: dataset processing, convex decomp, scene XML."""

    clip_root: Path
    """v2s2a clip output root containing the per-stage outputs
    (``outputs/<clip>/{process,hand_recon,segment,geometry,pose_estimation}``)."""

    task: str | None = None
    """Pipeline task name; defaults to the clip directory name."""

    output_root: Path | None = None
    """Retargeting output root; defaults to ``<clip_root>/scene_construction``.
    The do-as-i-do ``outputs/`` layout (``{robot}/{hand}/{task}/{data_id}`` +
    ``assets/``) is written here; the downstream retarget/physics_opt stages
    must be pointed at the same directory."""

    hand_type: Literal["auto", "left", "right", "bimanual"] = "auto"
    """Hand embodiment; ``auto`` resolves from the hand_recon valid masks."""

    robot_type: str = "sharpa"
    """Target robot hand directory under the repo's ``assets/hands/``."""

    data_id: int = 0
    """Trial index under the task directory."""

    dataset_name: str = "do_as_i_do"
    """Dataset tag recorded in task_info.json."""

    add_ur3_arm: bool = True
    """Add massless cylinders approximating the UR3e wrist housing."""

    force: bool = True
    """Overwrite existing stage outputs."""

    gravity: Literal["auto", "up", "json"] = "auto"
    """Gravity source: ``auto`` runs GeoCalib on the clip frames; ``up``
    assumes the camera Z axis is already up; ``json`` reads
    ``--gravity-json``."""

    gravity_json: Path | None = None
    """Precomputed gravity JSON (``vec3d``/``roll_deg``/``pitch_deg``)."""

    gravity_cache: bool = True
    """Cache the auto gravity estimate under ``<clip>/scene_construction/``."""

    gravity_camera_model: str = "pinhole"
    """GeoCalib camera model for the gravity estimate."""

    gravity_max_frames: int = 32
    """Subsample cap for the gravity estimate."""

    gravity_device: str = "cuda"
    """Torch device for the gravity estimate."""

    object_trajectory: Literal["auto", "canonical", "aligned"] = "auto"
    """Object pose source. ``auto`` uses an accepted optional alignment and
    otherwise preserves canonical pose_estimation behavior; ``canonical`` always
    ignores alignment; ``aligned`` requires an accepted alignment."""

    alignment_manifest: Path | None = None
    """Freely select an alignment poses.json. The default is
    ``<clip-root>/hand_object_alignment/poses.json``."""


def _resolve_gravity(args: SceneBuildArgs, frames_dir: Path, clip_root: Path):
    if args.gravity == "up":
        vec = np.array([0.0, 0.0, 1.0])
        return vec, {"roll_deg": 0.0, "pitch_deg": 0.0, "source": "up"}

    if args.gravity == "json":
        if args.gravity_json is None:
            raise ValueError("--gravity json requires --gravity-json <path>")
        data = json.loads(args.gravity_json.read_text(encoding="utf-8"))
        return np.asarray(data["vec3d"], dtype=np.float64), {**data, "source": "json"}

    cache_path = clip_root / "scene_construction" / "gravity.json"
    if args.gravity_cache and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info("[scene_construction] gravity cache hit: {}", cache_path)
        return np.asarray(data["vec3d"], dtype=np.float64), {**data, "source": "cache"}

    estimate = estimate_gravity_from_frames(
        frames_dir,
        camera_model=args.gravity_camera_model,
        max_frames=args.gravity_max_frames,
        device=args.gravity_device,
    )
    meta = {
        "roll_deg": estimate.roll_deg,
        "pitch_deg": estimate.pitch_deg,
        "n_frames": estimate.n_frames,
        "n_inliers": estimate.n_inliers,
        "source": "auto",
    }
    if args.gravity_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"vec3d": estimate.vec3d, **meta}, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("[scene_construction] wrote gravity cache → {}", cache_path)
    return np.asarray(estimate.vec3d, dtype=np.float64), meta


def run_scene_build(args: SceneBuildArgs) -> str:
    """Run stages 1-3; returns the resolved pipeline task name."""

    args.clip_root = args.clip_root.expanduser().resolve()
    if not args.clip_root.is_dir():
        raise FileNotFoundError(f"--clip-root not found: {args.clip_root}")
    task = args.task or args.clip_root.name
    if args.output_root is None:
        args.output_root = args.clip_root / "scene_construction"

    # Probe stage manifests once so the CLI fails fast on missing inputs.
    inputs: ClipInputs = load_clip_inputs(
        args.clip_root,
        task=task,
        embodiment_type=args.hand_type,
        gravity_up_cam=np.array([0.0, 0.0, 1.0]),  # placeholder; resolved below
        object_trajectory=args.object_trajectory,
        alignment_manifest=args.alignment_manifest,
    )
    gravity_up_cam, gravity_meta = _resolve_gravity(
        args, inputs.frames_dir, args.clip_root
    )
    inputs = replace(
        inputs,
        gravity_up_cam=gravity_up_cam,
        gravity_meta=gravity_meta,
    )

    pipeline_task = process_dataset(
        clip_root=str(args.clip_root),
        output_root_dir=str(args.output_root),
        task=task,
        data_id=args.data_id,
        embodiment_type=inputs.embodiment_type,
        dataset_name=args.dataset_name,
        inputs=inputs,
        force=args.force,
    )
    if pipeline_task is None:
        raise RuntimeError(
            f"{args.dataset_name} processing failed (no task_name returned)"
        )

    decompose_mesh(
        task=pipeline_task,
        dataset_name=args.dataset_name,
        data_id=args.data_id,
        output_root_dir=str(args.output_root),
        embodiment_type=inputs.embodiment_type,
        thicken=0.002,
        dilate=0.002,
        force=args.force,
    )

    # Scene flags for the do_as_i_do dataset: the object rests on an auto-placed
    # pedestal (with a welded support) rather than directly on the floor.
    generate_scene(
        task=pipeline_task,
        dataset_name=args.dataset_name,
        data_id=args.data_id,
        output_root_dir=str(args.output_root),
        embodiment_type=inputs.embodiment_type,
        robot_type=args.robot_type,
        show_viewer=False,
        friction_scale=1.5,
        object_floor_collision=False,
        hand_floor_collision=False,
        use_pedestal=True,
        use_support=True,
        force=args.force,
        add_ur3_arm=args.add_ur3_arm,
    )
    logger.info(
        "[scene_construction] stages 1-3 complete: task={}, robot={}",
        pipeline_task,
        args.robot_type,
    )
    return pipeline_task
