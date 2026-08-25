"""Resolve pipeline artifacts and launch the selected viewer."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from vis.paths import (
    resolve_config_yaml,
    resolve_embodiment,
    resolve_mano_npz,
    resolve_object_mesh,
    resolve_raw_inputs,
    resolve_robot,
    resolve_run_dir,
    resolve_task,
    resolve_traj_npz,
)


def _required(path: Path | None, description: str) -> Path:
    if path is None or not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path.resolve()


def run_scene(args, clip_root: Path) -> None:
    task = resolve_task(clip_root)
    embodiment = args.scene.embodiment or resolve_embodiment(clip_root, task)
    robot = args.scene.robot
    if args.scene.run_dir is not None:
        run_dir = _required(args.scene.run_dir, "Run directory")
    else:
        robot = robot or resolve_robot(clip_root, embodiment, task, args.scene.data_id)
        run_dir = resolve_run_dir(
            clip_root, robot, embodiment, task, args.scene.data_id
        ).resolve()

    scene_path = _required(args.scene.scene or run_dir / "scene.xml", "scene.xml")
    trajectory_path = _required(
        args.scene.traj or resolve_traj_npz(run_dir), "trajectory NPZ"
    )
    default_ik = run_dir / "trajectory_kinematic.npz"
    ik_path = args.scene.ik or (default_ik if default_ik.exists() else None)
    if ik_path is not None and ik_path.resolve() == trajectory_path:
        ik_path = None
    mano_path = args.scene.mano or resolve_mano_npz(
        clip_root, embodiment, task, args.scene.data_id
    )
    if not mano_path.exists():
        mano_path = None

    mesh_path = args.scene.mesh
    if mesh_path is None:
        mesh_result = resolve_object_mesh(
            clip_root, clip_root / "scene_construction", task, None
        )
        mesh_path = mesh_result[0] if mesh_result else None

    config_path = resolve_config_yaml(run_dir)
    config = dict(OmegaConf.load(config_path)) if config_path else {}

    from vis.scene import run_scene_viewer

    run_scene_viewer(
        scene_path=scene_path,
        trajectory_path=trajectory_path,
        ik_path=ik_path,
        mano_path=mano_path,
        object_mesh_path=mesh_path,
        config=config,
        port=args.port,
        fps=args.fps,
        skip_warmup=args.skip_warmup,
    )


def run_raw(args, clip_root: Path) -> None:
    inputs = resolve_raw_inputs(
        clip_root,
        resolve_task(clip_root),
        meshes_npz=args.raw.meshes_npz,
        poses_json=args.raw.poses_json,
        geometry_json=args.raw.geometry_json,
        hands_json=args.raw.hands_json,
        mesh=args.raw.mesh,
    )
    if not any((inputs.meshes_npz, inputs.poses_json, inputs.geometry_json)):
        raise FileNotFoundError(
            f"No raw visualization artifacts found under {clip_root}"
        )

    from vis.raw import run_raw_viewer

    run_raw_viewer(inputs, args.port, args.fps, args.raw.max_points)
