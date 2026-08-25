"""Tests for the video-mode (frame-by-frame) geometry workflow."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from geometry.moge_model import MogeFrameResult, MogeModel
from geometry.workflow import (
    GeometryVideoArgs,
    GeometryVideoOutputs,
    run_video_geometry,
)


def build_process_layout(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path]:
    """Create a synthetic ``process`` stage output and return (frames.json, clip_root)."""

    clip_root = tmp_path / "0"
    frames_dir = clip_root / "process" / "frames"
    frames_dir.mkdir(parents=True)
    for index in range(frame_count):
        rgb = np.full((8, 10, 3), index * 40, dtype=np.uint8)
        from PIL import Image

        Image.fromarray(rgb).save(frames_dir / f"{index:06d}.png")

    frames_json = clip_root / "process" / "frames.json"
    frames_json.write_text(
        json.dumps(
            {
                "source_video": str(tmp_path / "clip.mp4"),
                "fps": 10.0,
                "width": 10,
                "height": 8,
                "format": "png",
                "frame_count": frame_count,
                "frames_dir": str(frames_dir),
                "entries": [
                    {
                        "index": index,
                        "frame_filename": f"{index:06d}.png",
                        "timestamp_sec": index / 10.0,
                    }
                    for index in range(frame_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return frames_json, tmp_path


class FakeMogeModel:
    """Duck-typed stand-in for MogeModel (no MoGe checkpoint needed)."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def infer_image(
        self,
        frame_rgb: np.ndarray,
        fov_x: float | None = None,
    ) -> MogeFrameResult:
        height, width = frame_rgb.shape[:2]
        self.calls.append((height, width))
        ys = np.arange(height, dtype=np.float32)[:, None]
        xs = np.arange(width, dtype=np.float32)[None, :]
        depth = 1.0 + 0.1 * xs + 0.05 * ys
        mask = np.ones((height, width), dtype=bool)
        points = np.stack(
            [xs * depth, ys * depth, np.broadcast_to(depth, (height, width))],
            axis=-1,
        )
        intrinsics = np.array(
            [[0.5, 0.0, 0.5], [0.0, 0.75, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        return MogeFrameResult(
            points=points,
            depth=depth,
            mask=mask,
            normal=None,
            intrinsics=intrinsics,
        )


def _patch_model(monkeypatch: pytest.MonkeyPatch, fake: FakeMogeModel) -> None:
    monkeypatch.setattr(MogeModel, "__new__", lambda cls, args: fake)


def test_run_video_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path)
    fake = FakeMogeModel()
    _patch_model(monkeypatch, fake)

    outputs = run_video_geometry(
        GeometryVideoArgs(frames_json=frames_json, output_root=tmp_path, vis=True),
    )

    assert isinstance(outputs, GeometryVideoOutputs)
    assert outputs.stage_dir == tmp_path / "0" / "geometry"
    assert outputs.frames_dir.exists()
    assert outputs.vis_dir is not None and outputs.vis_dir.exists()
    assert len(fake.calls) == 3

    frame_dirs = sorted(p.name for p in outputs.frames_dir.iterdir())
    assert frame_dirs == ["000000", "000001", "000002"]

    first = outputs.frames_dir / "000000"
    for name in ("depth.exr", "mask.png", "points.npy", "intrinsics.npy"):
        assert (first / name).exists(), name

    depth = cv2.imread(str(first / "depth.exr"), cv2.IMREAD_UNCHANGED)
    assert depth is not None and depth.dtype == np.float32
    points = np.load(first / "points.npy")
    assert points.shape == (8, 10, 3)

    intrinsics = np.load(first / "intrinsics.npy")
    # Denormalized: fx * W = 0.5 * 10, fy * H = 0.75 * 8.
    assert intrinsics[0, 0] == pytest.approx(5.0)
    assert intrinsics[1, 1] == pytest.approx(6.0)

    vis = sorted(p.name for p in outputs.vis_dir.glob("*.png"))
    assert vis == [
        "000000_depth_vis.png",
        "000001_depth_vis.png",
        "000002_depth_vis.png",
    ]

    manifest = json.loads(outputs.geometry_json_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "geometry"
    assert manifest["frame_count"] == 3
    assert manifest["processed_count"] == 3
    assert len(manifest["entries"]) == 3
    entry = manifest["entries"][0]
    assert entry["index"] == 0
    assert entry["frame_filename"] == "000000.png"
    assert entry["depth"] == "depth.exr"
    assert entry["points"] == "points.npy"
    assert entry["intrinsics"] == "intrinsics.npy"
    assert entry["mask"] == "mask.png"
    assert entry["ply"] is None
    assert entry["vis"] == "000000_depth_vis.png"

    config = json.loads(outputs.config_json_path.read_text(encoding="utf-8"))
    assert config["package"]["name"] == "geometry"
    assert config["geometry"]["version"] == "v3"


def test_run_video_geometry_max_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=5)
    fake = FakeMogeModel()
    _patch_model(monkeypatch, fake)

    outputs = run_video_geometry(
        GeometryVideoArgs(
            frames_json=frames_json,
            output_root=tmp_path,
            vis=False,
            max_frames=2,
        ),
    )

    assert len(fake.calls) == 2
    manifest = json.loads(outputs.geometry_json_path.read_text(encoding="utf-8"))
    assert manifest["processed_count"] == 2
    assert [entry["index"] for entry in manifest["entries"]] == [0, 1]


def test_run_video_geometry_negative_max_frames(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path)
    with pytest.raises(ValueError):
        run_video_geometry(
            GeometryVideoArgs(frames_json=frames_json, max_frames=-1),
        )
