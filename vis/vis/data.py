"""Load and validate visualization artifacts without starting a server."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_qpos(path: Path, model_nq: int) -> np.ndarray:
    """Load a kinematic or chunked MJWP trajectory as ``(T, nq)``."""
    with np.load(path, allow_pickle=False) as archive:
        if "qpos" not in archive.files:
            raise ValueError(f"'qpos' not found in {path}; keys={archive.files}")
        qpos = np.asarray(archive["qpos"])
        if qpos.ndim == 3:
            if "sim_step" in archive.files:
                sim_step = np.asarray(archive["sim_step"]).reshape(-1)
                if len(sim_step) == qpos.shape[0]:
                    qpos = qpos[np.argsort(sim_step)]
            qpos = qpos.reshape(-1, qpos.shape[-1])

    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos shape (T, nq), got {qpos.shape} in {path}")
    if qpos.shape[1] != model_nq:
        raise ValueError(
            f"qpos width {qpos.shape[1]} != scene model.nq {model_nq}; "
            "the scene and trajectory likely belong to different runs"
        )
    return np.ascontiguousarray(qpos, dtype=np.float64)


def resolve_manifest_path(manifest_path: Path, value: str | Path) -> Path:
    """Resolve artifact paths written as either absolute or manifest-relative."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = (manifest_path.parent / path, Path.cwd() / path)
    return next(
        (candidate for candidate in candidates if candidate.exists()), candidates[0]
    )


@dataclass(frozen=True)
class RawFrame:
    index: int
    image_path: Path | None = None
    points_path: Path | None = None
    intrinsics_path: Path | None = None
    pose_path: Path | None = None


def load_raw_frames(
    geometry_json: Path | None, poses_json: Path | None
) -> list[RawFrame]:
    """Join geometry and pose manifests by pipeline frame index."""
    rows: dict[int, dict[str, Path | None]] = {}

    if geometry_json is not None:
        geometry = load_json(geometry_json)
        source_frames_path = geometry.get("source_frames_json")
        images: dict[int, Path] = {}
        if source_frames_path:
            frames_json = resolve_manifest_path(geometry_json, source_frames_path)
            if frames_json.exists():
                frame_manifest = load_json(frames_json)
                frames_dir = resolve_manifest_path(
                    frames_json, frame_manifest.get("frames_dir", "frames")
                )
                images = {
                    int(entry["index"]): frames_dir / entry["frame_filename"]
                    for entry in frame_manifest.get("entries", [])
                }
        for entry in geometry.get("entries", []):
            index = int(entry["index"])
            frame_dir = resolve_manifest_path(geometry_json, entry["frame_dir"])
            rows[index] = {
                "image_path": images.get(index),
                "points_path": frame_dir / entry.get("points", "points.npy"),
                "intrinsics_path": frame_dir
                / entry.get("intrinsics", "intrinsics.npy"),
                "pose_path": None,
            }

    if poses_json is not None:
        poses = load_json(poses_json)
        poses_dir = poses_json.parent / "poses"
        for entry in poses.get("entries", []):
            index = int(entry["index"])
            row = rows.setdefault(
                index,
                {
                    "image_path": None,
                    "points_path": None,
                    "intrinsics_path": None,
                    "pose_path": None,
                },
            )
            filename = entry.get("pose_filename")
            if filename and entry.get("tracked", True):
                row["pose_path"] = poses_dir / filename

    return [RawFrame(index=index, **rows[index]) for index in sorted(rows)]


def subsample_point_map(
    points: np.ndarray, colors: np.ndarray | None, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a point map, remove invalid points, and deterministically thin it."""
    flat_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(flat_points).all(axis=1) & (flat_points[:, 2] > 0)
    flat_points = flat_points[valid]
    if colors is None:
        flat_colors = np.full((len(flat_points), 3), 180, dtype=np.uint8)
    else:
        flat_colors = np.asarray(colors).reshape(-1, 3)[valid].astype(np.uint8)
    if max_points > 0 and len(flat_points) > max_points:
        selected = np.linspace(0, len(flat_points) - 1, max_points, dtype=np.int64)
        flat_points = flat_points[selected]
        flat_colors = flat_colors[selected]
    return flat_points, flat_colors
