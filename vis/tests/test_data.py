import json
from pathlib import Path

import numpy as np
import pytest

from vis.data import load_qpos, load_raw_frames, subsample_point_map


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_qpos_sorts_and_flattens_chunks(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.npz"
    qpos = np.array([[[20, 21], [22, 23]], [[10, 11], [12, 13]]])
    np.savez(path, qpos=qpos, sim_step=np.array([2, 1]))

    loaded = load_qpos(path, model_nq=2)

    np.testing.assert_array_equal(
        loaded, np.array([[10, 11], [12, 13], [20, 21], [22, 23]])
    )


def test_load_qpos_rejects_scene_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.npz"
    np.savez(path, qpos=np.zeros((2, 3)))

    with pytest.raises(ValueError, match="model.nq"):
        load_qpos(path, model_nq=4)


def test_load_raw_frames_joins_pipeline_manifests(tmp_path: Path) -> None:
    clip = tmp_path / "outputs" / "demo"
    frames_dir = clip / "process" / "frames"
    frames_dir.mkdir(parents=True)
    _write_json(
        clip / "process" / "frames.json",
        {
            "frames_dir": str(frames_dir),
            "entries": [{"index": 4, "frame_filename": "000004.png"}],
        },
    )
    geometry_dir = clip / "geometry" / "frames" / "000004"
    geometry_dir.mkdir(parents=True)
    geometry_json = clip / "geometry" / "geometry.json"
    _write_json(
        geometry_json,
        {
            "source_frames_json": str(clip / "process" / "frames.json"),
            "entries": [
                {
                    "index": 4,
                    "frame_dir": str(geometry_dir),
                    "points": "points.npy",
                    "intrinsics": "intrinsics.npy",
                }
            ],
        },
    )
    poses_json = clip / "pose_estimation" / "poses.json"
    _write_json(
        poses_json,
        {"entries": [{"index": 4, "pose_filename": "000004.txt", "tracked": True}]},
    )

    frames = load_raw_frames(geometry_json, poses_json)

    assert frames[0].index == 4
    assert frames[0].image_path == frames_dir / "000004.png"
    assert frames[0].points_path == geometry_dir / "points.npy"
    assert frames[0].intrinsics_path == geometry_dir / "intrinsics.npy"
    assert frames[0].pose_path == poses_json.parent / "poses" / "000004.txt"


def test_subsample_point_map_filters_invalid_and_is_bounded() -> None:
    points = np.array([[[0, 0, 1], [1, 0, np.inf], [2, 0, 2], [3, 0, 3]]])
    colors = np.arange(12, dtype=np.uint8).reshape(1, 4, 3)

    sampled_points, sampled_colors = subsample_point_map(points, colors, 2)

    np.testing.assert_array_equal(sampled_points, [[0, 0, 1], [3, 0, 3]])
    np.testing.assert_array_equal(sampled_colors, [colors[0, 0], colors[0, 3]])
