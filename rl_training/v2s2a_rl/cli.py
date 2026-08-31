"""User-facing prepare/train/eval entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from v2s2a_rl.bundle import TaskBundle, prepare_task_bundle
from v2s2a_rl.config import set_runtime_paths
from v2s2a_rl.conversion import convert_scene_mjcf

TASK = "V2S2A-Trajectory-Hand-v0"
PLAY_TASK = "V2S2A-Trajectory-Hand-Play-v0"


def _prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an RL task bundle from v2s2a outputs")
    parser.add_argument("--trajectory", type=Path, required=True, help="trajectory_kinematic.npz")
    parser.add_argument("--scene", type=Path, required=True, help="scene.xml with robot/object/scene assets")
    parser.add_argument("--keypoints", type=Path, default=None, help="trajectory_keypoints.npz")
    parser.add_argument("--output", type=Path, required=True, help="output task_bundle.json")
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-convert", action="store_true", help="skip MJCF-to-USD conversion")
    parser.add_argument("--force-convert", action="store_true")
    return parser


def prepare_main(argv: list[str] | None = None) -> None:
    args = _prepare_parser().parse_args(argv)
    scene_usd = None
    if args.no_convert and args.output.exists():
        scene_usd = TaskBundle.from_json(args.output).scene_usd_path
    elif not args.no_convert:
        scene_usd = convert_scene_mjcf(
            args.scene,
            args.output.expanduser().resolve().parent / "scene_usd",
            force=args.force_convert,
        )
    bundle = prepare_task_bundle(
        args.trajectory, args.scene, args.output, args.name, scene_usd, args.keypoints
    )
    print(f"Prepared {bundle.name}: {bundle.num_frames} frames, {bundle.hand_dofs} hand DoFs")
    print(args.output.expanduser().resolve())


def _runner(command: str, args: argparse.Namespace, passthrough: list[str]) -> int:
    bundle = args.bundle.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    set_runtime_paths(bundle, output)

    # IsaacLab pip wheels intentionally do not ship standalone runner scripts.
    # Our thin runners use the public IsaacLab/RSL-RL APIs and register this task.
    module = "v2s2a_rl.runners.train" if command == "train" else "v2s2a_rl.runners.play"
    task = TASK if command == "train" else PLAY_TASK
    cmd = [sys.executable, "-m", module, f"--task={task}"]
    if args.num_envs is not None:
        cmd += ["--num_envs", str(args.num_envs)]
    if command == "train" and args.max_iterations is not None:
        cmd += ["--max_iterations", str(args.max_iterations)]
    if command == "eval":
        cmd += ["--checkpoint", str(args.checkpoint.expanduser().resolve())]
        if args.video:
            cmd += ["--video", "--video_length", str(args.video_length)]
    cmd += passthrough
    return subprocess.call(cmd, cwd=output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="v2s2a-rl")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="validate inputs and write task_bundle.json")
    prepare.add_argument("--trajectory", type=Path, required=True)
    prepare.add_argument("--scene", type=Path, required=True)
    prepare.add_argument("--keypoints", type=Path, default=None)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--name", default=None)
    prepare.add_argument("--no-convert", action="store_true")
    prepare.add_argument("--force-convert", action="store_true")

    for name in ("train", "eval"):
        run = sub.add_parser(name, help=f"{name} with the official IsaacLab RSL-RL runner")
        run.add_argument("--bundle", type=Path, required=True)
        run.add_argument("--output-dir", type=Path, required=True)
        run.add_argument("--num-envs", type=int, default=None)
        if name == "train":
            run.add_argument("--max-iterations", type=int, default=None)
        else:
            run.add_argument("--checkpoint", type=Path, required=True)
            run.add_argument("--video", action="store_true")
            run.add_argument("--video-length", type=int, default=500)

    args, passthrough = parser.parse_known_args(argv)
    if args.command == "prepare":
        scene_usd = None
        if args.no_convert and args.output.exists():
            scene_usd = TaskBundle.from_json(args.output).scene_usd_path
        elif not args.no_convert:
            scene_usd = convert_scene_mjcf(
                args.scene,
                args.output.expanduser().resolve().parent / "scene_usd",
                force=args.force_convert,
            )
        bundle = prepare_task_bundle(
            args.trajectory, args.scene, args.output, args.name, scene_usd, args.keypoints
        )
        print(f"Prepared {bundle.name}: {bundle.num_frames} frames, {bundle.hand_dofs} hand DoFs")
        return
    raise SystemExit(_runner(args.command, args, passthrough))


if __name__ == "__main__":
    main()
