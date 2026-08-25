"""Tests for loading the process frame manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pose_estimation.frames import FrameManifest, load_frame_manifest


def build_process_layout(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path]:
    """Create a synthetic ``process`` stage output and return (frames.json, root)."""

    clip_root = tmp_path / "0"
    frames_dir = clip_root / "process" / "frames"
    frames_dir.mkdir(parents=True)
    for index in range(frame_count):
        rgb = np.full((8, 10, 3), index * 40, dtype=np.uint8)
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


def test_load_frame_manifest(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path)
    manifest = load_frame_manifest(frames_json)

    assert isinstance(manifest, FrameManifest)
    assert manifest.frame_count == 3
    assert manifest.fps == 10.0
    assert len(manifest.entries) == 3
    assert manifest.entries[0].path == (manifest.frames_dir / "000000.png").resolve()


def test_load_frame_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_frame_manifest(tmp_path / "missing.json")
