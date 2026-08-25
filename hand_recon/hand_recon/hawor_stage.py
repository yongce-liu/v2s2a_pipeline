"""Thin wrappers that adapt HaWoR's top-level modules to a per-clip pipeline.

HaWoR ships a flat ``lib`` / ``hawor`` / ``infiller`` / ``scripts`` /
``thirdparty`` layout that only imports correctly when those directories sit
at ``sys.path`` root and certain relative-path expectations (``./weights``,
``./_DATA/``, ``thirdparty/...``) are met. The ``hawor`` package built from
``pkgs/HaWoR/pyproject.toml`` handles the former via a .pth hook; this module
sets up the latter by bind-mounting the v2s2a_pipeline workspace over the
expected paths with symlinks, then cleaning them up when the run completes.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


from hand_recon.paths import HAWOR_SOURCE
from hand_recon.symlink_tracker import symlink_tracker


def _insert_syspath(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


# ``haworpkg`` is shipped by the pkgs/HaWoR wheel; its .pth hooks run before
# sitepackages so nothing further is needed once the venv is active. We only
# bring the third-party subtrees up to sys.path because Metri3D's files use
# intra-package imports like ``from mono.utils... import`` with no package
# prefix, and DROID-SLAM expects the droid_slam namespace at path root.
_HAWOR_THIRDPARTY = HAWOR_SOURCE / "thirdparty" / "Metric3D"
_HAWOR_DROID = HAWOR_SOURCE / "thirdparty" / "DROID-SLAM"


def ensure_hawor_importable() -> None:
    """Make the HaWoR top-level modules resolvable at runtime.

    Idempotent. Adds the two third-party subtrees that don't satisfy the
    self-namespace conventions expected by the wheel-from-repo layout.

    The editable ``hawor`` package installation from pkgs/HaWoR already
    places its ``lib``, ``hawor``, ``infiller``, and ``scripts`` namespaces
    on ``sys.path``; we only need to add ``thirdparty/Metric3D`` (its files
    import intra-package modules like ``from mono.utils ...`` with no
    package prefix) and ``thirdparty/DROID-SLAM`` (used by the optional
    SLAM path).
    """
    _insert_syspath(_HAWOR_THIRDPARTY)
    _insert_syspath(_HAWOR_DROID)


@dataclass(frozen=True)
class HaworWorkspacePaths:
    """Where HaWoR outputs land after redirecting its hardcoded paths."""

    workspace_root: Path
    """Scratch directory that plays the role of HaWoR's in-tree CWD."""

    droid_slam_dir: Path
    metric3d_dir: Path

    @property
    def seq_parent(self) -> Path:
        return self.workspace_root / "__outputs__"


@contextmanager
def _mount_hawor_assets(workspace_root: Path) -> Iterator[None]:
    """Bind-mount the HaWoR assets/scripts subtrees HaWoR expects CWD-relative.

    HaWoR reads ``./weights/external/detector.pt``, ``./weights/hawor/...``,
    ``./_DATA/data``, ``thirdparty/...``, and so on, relative to its CWD. The
    workspace_root directory is set up with symlinks so those paths resolve to
    the real assets in ``pkgs/HaWoR``.
    """
    workspace_root.mkdir(parents=True, exist_ok=True)

    with symlink_tracker() as tracker:
        tracker.link(workspace_root / "weights", HAWOR_SOURCE / "weights")
        tracker.link(workspace_root / "_DATA", HAWOR_SOURCE / "_DATA")
        tracker.link(workspace_root / "thirdparty", HAWOR_SOURCE / "thirdparty")
        tracker.link(workspace_root / "scripts", HAWOR_SOURCE / "scripts")
        tracker.link(workspace_root / "lib", HAWOR_SOURCE / "lib")
        tracker.link(workspace_root / "hawor", HAWOR_SOURCE / "hawor")
        tracker.link(workspace_root / "infiller", HAWOR_SOURCE / "infiller")

        prev_cwd = Path.cwd()
        os.chdir(workspace_root)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


@dataclass
class HandReconHaworArgs:
    """Settings HaWoR's pipeline needs (crowded argparse-style params object)."""

    video_path: Path
    checkpoint: Path
    infiller_weight: Path
    img_focal: float | None = None
    static_camera: bool = True
    vis_mode: str = "cam"


@contextmanager
def hawor_workspace_env(
    hand_recon_root: Path,
) -> Iterator[HaworWorkspacePaths]:
    """Mount the assets HaWoR expects CWD-relative, then cd into the workspace.

    Caller still needs to ``ensure_hawor_importable()`` first.
    """
    workspace_root = hand_recon_root / ".hawor_runtime"
    with _mount_hawor_assets(workspace_root):
        yield HaworWorkspacePaths(
            workspace_root=workspace_root,
            droid_slam_dir=workspace_root / "thirdparty" / "DROID-SLAM",
            metric3d_dir=workspace_root / "thirdparty" / "Metric3D",
        )


def run_detect_track(
    args: HandReconHaworArgs,
    workspace: HaworWorkspacePaths,
) -> tuple[int, int, Path, list[str]]:
    """Run HaWoR's detect+track stage.

    The process stage has already extracted frames; pre-populate HaWoR's
    ``extracted_images/`` directory so ``detect_track_video`` skips its
    ffmpeg re-extraction and goes straight to YOLO detection.

    Returns ``(start_idx, end_idx, seq_folder, imgfiles)``.
    """

    from scripts.scripts_test_video.detect_track_video import detect_track_video

    seq_stem = _seq_stem(args)
    seq_folder = workspace.seq_parent / seq_stem
    extracted = seq_folder / "extracted_images"
    extracted.mkdir(parents=True, exist_ok=True)

    # Populate extracted_images from the source frames directory. HaWoR's
    # scripts glob ``%04d.jpg`` in this folder; process outputs ``%06d.png``
    # whose frame index matches the manifest's ``index`` field. Symlink each
    # under HaWoR's expected naming so nothing is copied.
    src_frames = sorted(args.video_path.glob("*.png")) + sorted(
        args.video_path.glob("*.jpg")
    )
    if not src_frames:
        raise FileNotFoundError(
            f"No frames found in {args.video_path}; run the process stage first."
        )
    for idx, frame_path in enumerate(src_frames):
        target = extracted / f"{idx:04d}.jpg"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(frame_path.resolve())

    # Place a dummy file entry for HaWoR's argparse plumbing; it is never
    # actually read because extracted_images is already populated.
    redirect_video = workspace.workspace_root / "__outputs__" / f"{seq_stem}.mp4"
    redirect_video.parent.mkdir(parents=True, exist_ok=True)
    if redirect_video.is_symlink() or redirect_video.exists():
        redirect_video.unlink()
    redirect_video.touch()

    inner_args = _hawor_argparse_args(args, redirect_video)
    start_idx, end_idx, seq_folder_out, imgfiles = detect_track_video(inner_args)
    return start_idx, end_idx, Path(seq_folder_out), list(imgfiles)


def _seq_stem(args: HandReconHaworArgs) -> str:
    return args.video_path.stem if args.video_path.is_file() else args.video_path.name


def _redirect_video_path(
    workspace: HaworWorkspacePaths, args: HandReconHaworArgs
) -> Path:
    return workspace.workspace_root / "__outputs__" / f"{_seq_stem(args)}.mp4"


def run_motion_estimation(
    args: HandReconHaworArgs,
    workspace: HaworWorkspacePaths,
    start_idx: int,
    end_idx: int,
    seq_folder: Path,
) -> tuple[dict, float]:
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation

    inner_args = _hawor_argparse_args(args, _redirect_video_path(workspace, args))
    return hawor_motion_estimation(inner_args, start_idx, end_idx, str(seq_folder))


def run_slam(
    args: HandReconHaworArgs,
    workspace: HaworWorkspacePaths,
    start_idx: int,
    end_idx: int,
) -> tuple[object, object, object, object]:
    from lib.eval_utils.custom_utils import load_slam_cam
    from scripts.scripts_test_video.hawor_slam import hawor_slam

    inner_args = _hawor_argparse_args(args, _redirect_video_path(workspace, args))

    seq_folder = workspace.seq_parent / _seq_stem(args)
    slam_path = seq_folder / f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    if not slam_path.exists():
        hawor_slam(inner_args, start_idx, end_idx)
    return load_slam_cam(str(slam_path))


def run_infiller(
    args: HandReconHaworArgs,
    workspace: HaworWorkspacePaths,
    start_idx: int,
    end_idx: int,
    frame_chunks_all: dict,
) -> tuple:
    """Run HaWoR's infiller stage.

    Two upstream variants exist:

    - ``pkgs/HaWoR`` upstream: ``hawor_infiller(args, start, end, frame_chunks)``
      always expects the SLAM file on disk (unusable in --static-camera mode).
    - do-as-i-do patch: ``hawor_infiller(..., static_camera)`` that substitutes
      identity transforms when needed.

    Here we call the upstream signature and pre-create the SLAM .npz with
    identity transforms when ``static_camera`` is on, so the single upstream
    signature serves both modes without modifying pkgs/HaWoR.
    """
    from scripts.scripts_test_video.hawor_video import hawor_infiller

    if args.static_camera:
        _seed_identity_slam(workspace, args, start_idx, end_idx)

    inner_args = _hawor_argparse_args(args, _redirect_video_path(workspace, args))
    return hawor_infiller(inner_args, start_idx, end_idx, frame_chunks_all)


def _seed_identity_slam(
    workspace: HaworWorkspacePaths,
    args: HandReconHaworArgs,
    start_idx: int,
    end_idx: int,
) -> None:
    """Pre-create the SLAM npz with identity world<->camera transforms."""
    import numpy as np

    seq_folder = workspace.seq_parent / _seq_stem(args)
    slam_path = seq_folder / f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    if slam_path.exists():
        return

    extracted = seq_folder / "extracted_images"
    num_frames = len(list(extracted.glob("*.jpg")))

    extracted.mkdir(parents=True, exist_ok=True)
    slam_path.parent.mkdir(parents=True, exist_ok=True)

    identity = np.eye(4, dtype=np.float32)
    # 7-dim traj rows: tx ty tz qx qy qz qw. DROID-SLAM convention.
    traj = np.zeros((num_frames, 7), dtype=np.float32)
    traj[:, 6] = 1.0
    np.savez(
        slam_path,
        tstamp=np.arange(num_frames, dtype=np.int64),
        disps=np.ones((num_frames, 1, 1), dtype=np.float32),
        traj=traj,
        img_focal=args.img_focal if args.img_focal is not None else 600.0,
        img_center=np.array([640.0, 360.0], dtype=np.float32),
        scale=1.0,
    )


def _hawor_argparse_args(args: HandReconHaworArgs, redirect_video: Path):
    """Build the argparse-style object HaWoR's scripts expect."""

    return type(
        "HaworNamespace",
        (),
        {
            "video_path": str(redirect_video),
            "checkpoint": str(args.checkpoint),
            "infiller_weight": str(args.infiller_weight),
            "img_focal": args.img_focal,
            "static_camera": args.static_camera,
            "vis_mode": args.vis_mode,
            "input_type": "file",
        },
    )()
