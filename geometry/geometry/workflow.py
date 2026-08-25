"""MoGe geometry estimation: single image and frame-by-frame video modes.

Outputs (mirroring the ``segment`` stage layout):

.. code-block:: text

    outputs/<clip>/geometry/
    ├── config.json      # effective run config
    ├── geometry.json    # per-frame manifest (depth / points / intrinsics paths)
    └── frames/
        ├── 000000/
        │   ├── depth.exr        # float depth map
        │   ├── mask.png         # valid-pixel mask
        │   ├── points.npy       # camera-space point map (H, W, 3)
        │   ├── intrinsics.npy   # denormalized 3x3 camera intrinsics
        │   └── pointcloud.ply   # only with --save-ply
        └── ...
    └── vis/                 # colorized depth per frame, only with --vis
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

from geometry import (
    __version__,
    colorize_depth,
    load_rgb_image,
    save_depth_exr,
    save_mask,
    save_points_ply,
)
from geometry.frames import FrameManifest, load_frame_manifest
from geometry.moge_model import MogeArgs, MogeFrameResult, MogeModel

FRAME_DIRNAME_PATTERN = "{:06d}"


@dataclass
class GeometryVideoArgs:
    """Arguments for frame-by-frame geometry estimation of a whole video."""

    frames_json: Path
    """Path to the ``process`` stage's ``frames.json`` (the frame manifest)."""

    output_root: Path = Path(__file__).parents[2] / "outputs"
    """Root under which ``<clip_stem>/geometry/`` is created."""

    vis: bool = True
    """Write a colorized depth visualization per frame."""

    max_frames: int | None = None
    """Limit the number of frames processed (None = all frames in the manifest)."""

    save_ply: bool = False
    """Also write a colored ``pointcloud.ply`` per frame."""

    moge: MogeArgs = field(default_factory=MogeArgs)
    """MoGe model settings (checkpoint, version, resolution, ...)."""


@dataclass(frozen=True)
class GeometryEntry:
    """Per-frame record written into ``geometry.json``."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    frame_dir: str
    depth_filename: str
    points_filename: str
    intrinsics_filename: str
    mask_filename: str | None
    ply_filename: str | None
    vis_filename: str | None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frame_filename": self.frame_filename,
            "timestamp_sec": self.timestamp_sec,
            "frame_dir": self.frame_dir,
            "depth": self.depth_filename,
            "points": self.points_filename,
            "intrinsics": self.intrinsics_filename,
            "mask": self.mask_filename,
            "ply": self.ply_filename,
            "vis": self.vis_filename,
        }


@dataclass(frozen=True)
class GeometrySingleOutputs:
    """Everything produced by one single-image run."""

    output_dir: Path
    result: MogeFrameResult


@dataclass(frozen=True)
class GeometryVideoOutputs:
    """Everything produced by one video-mode run."""

    clip_root: Path
    stage_dir: Path
    frames_dir: Path
    vis_dir: Path | None
    geometry_json_path: Path
    config_json_path: Path
    entries: list[GeometryEntry]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _frame_outputs(
    result: MogeFrameResult,
    frame_rgb: np.ndarray,
    frame_dir: Path,
    args_moge: MogeArgs,
    save_ply: bool,
    vis_dir: Path | None,
    index: int,
) -> dict:
    """Write all on-disk artifacts for one frame and return its manifest entry."""

    height, width = frame_rgb.shape[:2]
    intrinsics = result.denormalized_intrinsics(width=width, height=height)

    frame_dir.mkdir(parents=True, exist_ok=True)
    save_depth_exr(result.depth, frame_dir / "depth.exr", args_moge.overwrite)
    np.save(frame_dir / "points.npy", result.points.astype(np.float32))
    np.save(frame_dir / "intrinsics.npy", intrinsics.astype(np.float64))
    if result.mask is not None:
        save_mask(
            (result.mask.astype(np.uint8)) * 255,
            frame_dir / "mask.png",
            args_moge.overwrite,
        )
    if save_ply:
        save_points_ply(
            result.points,
            frame_rgb,
            frame_dir / "pointcloud.ply",
            args_moge.overwrite,
        )

    vis_filename = None
    if vis_dir is not None:
        from PIL import Image

        vis_filename = f"{FRAME_DIRNAME_PATTERN.format(index)}_depth_vis.png"
        vis_path = vis_dir / vis_filename
        if not vis_path.exists() or args_moge.overwrite:
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(colorize_depth(result.depth)).save(vis_path)

    return {
        "depth_filename": "depth.exr",
        "points_filename": "points.npy",
        "intrinsics_filename": "intrinsics.npy",
        "mask_filename": "mask.png" if result.mask is not None else None,
        "ply_filename": (
            "pointcloud.ply" if (frame_dir / "pointcloud.ply").exists() else None
        ),
        "vis_filename": vis_filename,
    }


def process_single_image(
    image_path: Path,
    output_dir: Path,
    args: MogeArgs,
    model: MogeModel | None = None,
) -> GeometrySingleOutputs:
    """Run MoGe on one image and write its artifacts into ``output_dir``."""

    image_path = image_path.expanduser()
    output_dir = output_dir.expanduser()
    frame_rgb = load_rgb_image(image_path)
    active_model = model or MogeModel(args)
    result = active_model.infer_image(frame_rgb, fov_x=args.fov_x)

    payload = _frame_outputs(
        result=result,
        frame_rgb=frame_rgb,
        frame_dir=output_dir,
        args_moge=args,
        save_ply=True,
        vis_dir=output_dir / "vis",
        index=0,
    )
    _write_json(
        output_dir / "result.json",
        {
            "package": {"name": "geometry", "version": __version__},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_image": str(image_path.resolve()),
            "width": frame_rgb.shape[1],
            "height": frame_rgb.shape[0],
            **payload,
        },
    )
    return GeometrySingleOutputs(output_dir=output_dir, result=result)


def run_video_geometry(args: GeometryVideoArgs) -> GeometryVideoOutputs:
    """Estimate geometry for every frame in a ``process`` manifest."""

    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")

    frames_json = args.frames_json.expanduser().resolve()
    manifest = load_frame_manifest(frames_json)

    clip_stem = frames_json.parent.parent.name
    clip_root = args.output_root.expanduser().resolve() / clip_stem
    stage_dir = clip_root / "geometry"
    frames_out_dir = stage_dir / "frames"
    vis_dir = stage_dir / "vis" if args.vis else None

    if args.moge.overwrite:
        if frames_out_dir.exists():
            shutil.rmtree(frames_out_dir)
        if vis_dir is not None and vis_dir.exists():
            shutil.rmtree(vis_dir)
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    selected = manifest.entries
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    model = MogeModel(args.moge)
    entries: list[GeometryEntry] = []
    for frame in selected:
        frame_rgb = load_rgb_image(frame.path)
        result = model.infer_image(frame_rgb, fov_x=args.moge.fov_x)
        frame_dir = frames_out_dir / FRAME_DIRNAME_PATTERN.format(frame.index)
        files = _frame_outputs(
            result=result,
            frame_rgb=frame_rgb,
            frame_dir=frame_dir,
            args_moge=args.moge,
            save_ply=args.save_ply,
            vis_dir=vis_dir,
            index=frame.index,
        )
        entries.append(
            GeometryEntry(
                index=frame.index,
                frame_filename=frame.frame_filename,
                timestamp_sec=frame.timestamp_sec,
                frame_dir=str(frame_dir),
                **files,
            )
        )
        logger.debug(
            "[geometry] frame {}: depth={} points={}",
            frame.index,
            result.depth.shape,
            result.points.shape,
        )

    _write_json(
        stage_dir / "geometry.json",
        _manifest_dict(manifest, args, entries, frames_out_dir),
    )
    _write_json(stage_dir / "config.json", _config_dict(args, manifest))

    logger.info(
        "[geometry] Done: frames={} out={}",
        len(entries),
        stage_dir,
    )
    return GeometryVideoOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        frames_dir=frames_out_dir,
        vis_dir=vis_dir,
        geometry_json_path=stage_dir / "geometry.json",
        config_json_path=stage_dir / "config.json",
        entries=entries,
    )


def _manifest_dict(
    manifest: FrameManifest,
    args: GeometryVideoArgs,
    entries: list[GeometryEntry],
    frames_out_dir: Path,
) -> dict:
    return {
        "schema_version": "1.0",
        "stage": "geometry",
        "source_frames_json": str(args.frames_json.expanduser().resolve()),
        "source_video": manifest.source_video,
        "fps": manifest.fps,
        "width": manifest.width,
        "height": manifest.height,
        "frame_format": manifest.format,
        "frame_count": manifest.frame_count,
        "processed_count": len(entries),
        "frames_dir": str(frames_out_dir),
        "moge_version": args.moge.version,
        "entries": [entry.to_dict() for entry in entries],
    }


def _config_dict(args: GeometryVideoArgs, manifest: FrameManifest) -> dict:
    checkpoint = str(args.moge.checkpoint.expanduser())
    return {
        "package": {"name": "geometry", "version": __version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "frames_json": str(args.frames_json.expanduser().resolve()),
            "source_video": manifest.source_video,
            "fps": manifest.fps,
            "width": manifest.width,
            "height": manifest.height,
            "frame_format": manifest.format,
            "frame_count": manifest.frame_count,
        },
        "geometry": {
            "checkpoint": checkpoint,
            "allow_hf_download": args.moge.allow_hf_download,
            "device": args.moge.device,
            "version": args.moge.version,
            "fov_x": args.moge.fov_x,
            "resolution_level": args.moge.resolution_level,
            "num_tokens": args.moge.num_tokens,
            "refine_steps": args.moge.refine_steps,
            "use_fp16": args.moge.use_fp16,
            "force_projection": args.moge.force_projection,
            "apply_mask": args.moge.apply_mask,
            "vis": args.vis,
            "save_ply": args.save_ply,
            "overwrite": args.moge.overwrite,
            "max_frames": args.max_frames,
        },
        "software": {},
    }
