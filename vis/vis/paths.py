"""Pure path/artifact resolution for the vis CLI (no viser/mujoco imports).

All inputs are optional and keyed off the clip root ``outputs/<clip>/``; every
file can also be pinned with an explicit override from the CLI.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


def resolve_task(clip_root: Path) -> str:
    """Task name = clip directory name."""
    return clip_root.name


def resolve_embodiment(clip_root: Path, task: str) -> str:
    """Glob ``scene_construction/mano/*/<task>`` for the embodiment (one dir)."""
    pattern = str(clip_root / "scene_construction" / "mano" / "*" / task)
    embodiments = sorted(
        {Path(d).parent.name for d in glob.glob(pattern) if Path(d).is_dir()}
    )
    if not embodiments:
        raise FileNotFoundError(
            f"No scene_construction/mano/*/{task} under {clip_root}; "
            "run scene_construction first (or pass --embodiment)."
        )
    if len(embodiments) > 1:
        raise RuntimeError(
            f"Task '{task}' found multiple embodiments {embodiments}; pass --embodiment."
        )
    return embodiments[0]


def resolve_robot(clip_root: Path, embodiment: str, task: str, data_id: str) -> str:
    """Glob ``scene_construction/*/<embodiment>/<task>/<data_id>/scene.xml``."""
    pattern = str(
        clip_root
        / "scene_construction"
        / "*"
        / embodiment
        / task
        / data_id
        / "scene.xml"
    )
    robots = sorted({Path(d).parents[3].name for d in glob.glob(pattern)})
    if not robots:
        raise FileNotFoundError(
            f"No scene.xml matching scene_construction/*/{embodiment}/{task}/{data_id} "
            f"under {clip_root}; run retarget first (or pass --robot/--run-dir)."
        )
    if len(robots) > 1:
        raise RuntimeError(
            f"Multiple robots {robots} match; pass --robot or --run-dir."
        )
    return robots[0]


def resolve_run_dir(
    clip_root: Path,
    robot: str,
    embodiment: str,
    task: str,
    data_id: str,
) -> Path:
    return clip_root / "scene_construction" / robot / embodiment / task / data_id


def resolve_mano_npz(clip_root: Path, embodiment: str, task: str, data_id: str) -> Path:
    return (
        clip_root
        / "scene_construction"
        / "mano"
        / embodiment
        / task
        / data_id
        / "trajectory_keypoints.npz"
    )


def resolve_traj_npz(run_dir: Path) -> Path | None:
    """physics_opt trajectory; falls back to the IK trajectory when the physics
    stage has not run yet."""
    return _first_existing(
        [
            run_dir / "physics_opt" / "trajectory_mjwp.npz",
            run_dir / "trajectory_kinematic.npz",
        ]
    )


def resolve_config_yaml(run_dir: Path) -> Path | None:
    return _first_existing(
        [
            run_dir / "physics_opt" / "config.yaml",
            run_dir / "config.yaml",
        ]
    )


def resolve_object_mesh(
    clip_root: Path,
    output_root: Path,
    task: str,
    object_name: str | None,
) -> tuple[Path, float] | None:
    """(mesh_path, metric_scale) or None.

    Prefers the scene_construction asset copy ``assets/objects/<obj>/visual.obj``
    (already metric); falls back to the obj_recon reconstruction under
    ``obj_recon/meshes/{mv,*}/<obj>/<obj>.obj`` (canonical units, scaled by
    ``mean(local_to_scene.scale)`` from the sibling ``layout.json``).
    """
    # scene_construction asset copy (metric, includes convex/ siblings).
    obj = object_name or task
    asset_copy = output_root / "assets" / "objects" / obj / "visual.obj"
    if asset_copy.exists():
        return asset_copy, 1.0

    # obj_recon reconstruction (canonical units — needs layout.json scale).
    recon_root = clip_root / "obj_recon" / "meshes"
    meshes = sorted(recon_root.glob(f"**/{obj}/{obj}.obj")) or sorted(
        recon_root.glob(f"**/{obj}.obj")
    )
    for mesh in meshes:
        scale = 1.0
        layout = mesh.parent / "layout.json"
        if layout.exists():
            try:
                payload = json.loads(layout.read_text(encoding="utf-8"))
                objs = payload.get("objects", [])
                s = next(
                    (
                        o["local_to_scene"]["scale"]
                        for o in objs
                        if Path(o.get("mesh_obj", "")).name == mesh.name
                    ),
                    objs[0]["local_to_scene"]["scale"] if objs else None,
                )
                if s:
                    scale = float(sum(s) / len(s))
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                logger.warning(f"Could not read scale from {layout}: {e} — scale=1")
        return mesh, scale
    return None


@dataclass
class RawInputs:
    """Everything raw (camera-frame) mode needs; fields may be None."""

    meshes_npz: Path | None
    poses_json: Path | None
    geometry_json: Path | None
    hands_json: Path | None
    mesh_path: Path | None
    mesh_scale: float = 1.0


def resolve_raw_inputs(
    clip_root: Path,
    task: str,
    meshes_npz: Path | None = None,
    poses_json: Path | None = None,
    geometry_json: Path | None = None,
    hands_json: Path | None = None,
    mesh: Path | None = None,
) -> RawInputs:
    meshes_npz = meshes_npz or _first_existing(
        [clip_root / "hand_recon" / "meshes.npz"]
    )
    hands_json = hands_json or _first_existing(
        [clip_root / "hand_recon" / "hands.json"]
    )
    poses_json = poses_json or _first_existing(
        [clip_root / "pose_estimation" / "poses.json"]
    )
    geometry_json = geometry_json or _first_existing(
        [clip_root / "geometry" / "geometry.json"]
    )
    if mesh is None:
        found = resolve_object_mesh(
            clip_root, clip_root / "scene_construction", task, None
        )
        mesh_path, mesh_scale = found if found else (None, 1.0)
    else:
        mesh_path, mesh_scale = mesh, 1.0
    return RawInputs(
        meshes_npz=meshes_npz,
        poses_json=poses_json,
        geometry_json=geometry_json,
        hands_json=hands_json,
        mesh_path=mesh_path,
        mesh_scale=mesh_scale,
    )
