"""Tests for consuming geometry-stage point maps."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from obj_recon.reconstruct import (
    _layout_has_objects,
    _load_intrinsics,
    _load_pointmap,
    load_geometry_manifest,
)


def test_load_geometry_manifest_resolves_entry_paths(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frame_dir = frames_dir / "000007"
    frame_dir.mkdir(parents=True)
    manifest_path = tmp_path / "geometry.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "geometry",
                "frames_dir": str(frames_dir),
                "entries": [
                    {
                        "index": 7,
                        "frame_filename": "000007.png",
                        "frame_dir": str(frame_dir),
                        "points": "points.npy",
                        "intrinsics": "intrinsics.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    frames = load_geometry_manifest(manifest_path)

    assert frames[7].frame_filename == "000007.png"
    assert frames[7].points_path == (frame_dir / "points.npy").resolve()
    assert frames[7].intrinsics_path == (frame_dir / "intrinsics.npy").resolve()


def test_load_pointmap_converts_moge_camera_coordinates(tmp_path: Path):
    points_path = tmp_path / "points.npy"
    np.save(points_path, np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32))

    pointmap = _load_pointmap(points_path, (1, 1))

    assert pointmap.dtype == torch.float32
    assert torch.equal(pointmap, torch.tensor([[[-1.0, -2.0, 3.0]]]))


def test_load_pointmap_rejects_frame_shape_mismatch(tmp_path: Path):
    points_path = tmp_path / "points.npy"
    np.save(points_path, np.zeros((2, 3, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Point map shape mismatch"):
        _load_pointmap(points_path, (3, 2))


def test_load_intrinsics_normalizes_pixel_units(tmp_path: Path):
    intrinsics_path = tmp_path / "intrinsics.npy"
    np.save(
        intrinsics_path,
        np.array([[500.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]),
    )

    intrinsics = _load_intrinsics(intrinsics_path, (480, 640))

    assert torch.allclose(
        intrinsics,
        torch.tensor([[0.78125, 0.0, 0.5], [0.0, 0.8333333, 0.5], [0.0, 0.0, 1.0]]),
    )


def test_layout_has_objects_requires_nonempty_layout(tmp_path: Path):
    layout_path = tmp_path / "layout.json"
    layout_path.write_text('{"objects": []}', encoding="utf-8")
    assert not _layout_has_objects(layout_path)

    layout_path.write_text(
        '{"objects": [{"mesh_obj": "object.obj"}]}', encoding="utf-8"
    )
    assert _layout_has_objects(layout_path)
