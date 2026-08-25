"""Tests for the pose_estimation workflow manifest helpers (no GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pose_estimation.workflow import (
    PoseEstimationVideoArgs,
    _depth_from_geometry,
    _find_layout_path,
    _geometry_frame_dir,
    _load_mv_anchor_poses,
    _mask_path,
    _resolve_anchor_frames,
    load_geometry_manifest,
    load_masks_manifest,
)


def test_load_geometry_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_geometry_manifest(tmp_path / "missing.json")


def test_load_masks_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_masks_manifest(tmp_path / "missing.json")


def test_geometry_frame_dir_lookup() -> None:
    manifest = {
        "entries": [
            {"index": 0, "frame_dir": "/tmp/a/000000"},
            {"index": 1, "frame_dir": "/tmp/a/000001"},
        ]
    }
    assert _geometry_frame_dir(manifest, 1) == Path("/tmp/a/000001")
    assert _geometry_frame_dir(manifest, 5) is None


def test_mask_path_lookup(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    (masks_dir / "000000.png").write_bytes(b"")

    manifest = {
        "masks_dir": str(masks_dir),
        "entries": [
            {"index": 0, "mask_filename": "000000.png", "has_mask": True},
            {"index": 1, "mask_filename": "000001.png", "has_mask": True},
            {"index": 2, "mask_filename": None, "has_mask": False},
        ],
    }
    assert _mask_path(manifest, 0) == masks_dir / "000000.png"
    assert _mask_path(manifest, 1) is None  # file missing on disk
    assert _mask_path(manifest, 2) is None  # has_mask False


def test_mask_path_lookup_by_prompt(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"
    object_dir = masks_dir / "yellow spoon"
    object_dir.mkdir(parents=True)
    (object_dir / "000000.png").write_bytes(b"")
    manifest = {
        "masks_dir": str(masks_dir),
        "entries": [
            {
                "index": 0,
                "mask_filename": "000000.png",
                "has_mask": True,
                "prompt_masks": [
                    {
                        "prompt_id": "yellow spoon",
                        "mask_filename": "yellow spoon/000000.png",
                        "has_mask": True,
                    }
                ],
            }
        ],
    }
    assert _mask_path(manifest, 0, "yellow spoon") == object_dir / "000000.png"
    assert _mask_path(manifest, 0, "missing") is None


def test_find_layout_path_prefers_mesh_directory(tmp_path: Path) -> None:
    mesh_dir = tmp_path / "meshes" / "mv" / "object"
    mesh_dir.mkdir(parents=True)
    mesh_path = mesh_dir / "object.obj"
    mesh_path.write_text("")
    per_object = mesh_dir / "layout.json"
    parent = mesh_dir.parent / "layout.json"
    per_object.write_text("{}")
    parent.write_text("{}")
    assert _find_layout_path(mesh_path) == per_object


def test_resolve_mv_anchor_frames(tmp_path: Path) -> None:
    object_dir = tmp_path / "obj_recon" / "meshes" / "mv" / "yellow_spoon"
    object_dir.mkdir(parents=True)
    mesh_path = (object_dir / "yellow_spoon.obj").resolve()
    mesh_path.write_text("")
    (object_dir / "view_poses.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "stage": "obj_recon",
                "coordinate_frame": "pytorch3d_camera",
                "pose_convention": "object_local_to_camera",
                "views": [
                    {
                        "frame_index": index,
                        "scale": [0.2, 0.2, 0.2],
                        "object_to_camera_opencv": np.eye(4).tolist(),
                        "reference": index == 44,
                    }
                    for index in [44, 0, 88]
                ],
            }
        )
    )
    anchors = _load_mv_anchor_poses(mesh_path)
    assert [anchor.frame_index for anchor in anchors] == [44, 0, 88]
    assert _resolve_anchor_frames(mesh_path, None) == [44]
    assert _resolve_anchor_frames(mesh_path, [0, 88]) == [44]


def test_mv_pose_contract_is_required(tmp_path: Path) -> None:
    object_dir = tmp_path / "obj_recon" / "meshes" / "mv" / "object"
    object_dir.mkdir(parents=True)
    mesh_path = object_dir / "object.obj"
    mesh_path.write_text("")
    with pytest.raises(FileNotFoundError, match="MV pose contract"):
        _resolve_anchor_frames(mesh_path, None)


def test_depth_from_geometry(tmp_path: Path) -> None:
    points = np.ones((4, 5, 3), dtype=np.float32)
    points[..., 2] = 0.5
    points[0, 0, 2] = np.inf
    points[0, 1, 2] = 0.0

    np.save(tmp_path / "points.npy", points)
    depth = _depth_from_geometry(tmp_path)

    assert depth.shape == (4, 5)
    assert depth[0, 0] == 0.0  # inf zeroed
    assert depth[0, 1] == 0.0  # below threshold zeroed
    assert depth[1, 0] == pytest.approx(0.5)


def test_video_args_defaults() -> None:
    args = PoseEstimationVideoArgs(
        frames_json=Path("frames.json"),
        mesh_path=Path("mesh.obj"),
        masks_json=Path("masks.json"),
    )
    assert args.init_frame == 0
    assert args.anchor_frames is None
    assert args.prompt_id is None
    assert args.reinit_every == 0
    assert args.max_frames is None
    assert args.foundationpose.est_refine_iter == 5
    assert args.foundationpose.track_refine_iter == 10
    assert args.foundationpose.track_crop_ratio == 2.0
