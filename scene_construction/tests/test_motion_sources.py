"""Unit tests for the v2s2a stage-manifest loaders in motion_sources."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scene_construction.motion_sources import (
    _load_object_trajectory,
    estimate_mesh_scale,
    load_clip_inputs,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_clip(tmp_path: Path, n_frames: int = 4) -> Path:
    clip = tmp_path / "clip_a"

    poses_dir = clip / "pose_estimation" / "poses"
    poses_dir.mkdir(parents=True)
    for i in range(n_frames):
        pose = np.eye(4)
        pose[:3, 3] = [0.01 * i, 0.0, 0.5]
        np.savetxt(poses_dir / f"{i:06d}.txt", pose)
    _write(
        clip / "pose_estimation" / "poses.json",
        {
            "mesh_path": str(
                clip / "obj_recon" / "meshes" / "000000" / "thing" / "thing.obj"
            ),
            "object_name": "thing",
            "entries": [
                {
                    "index": i,
                    "tracked": True,
                    "pose_filename": f"{i:06d}.txt",
                }
                for i in range(n_frames)
            ],
        },
    )

    # Minimal mesh with extent 1.0 in x (raw units); needs a face for
    # trimesh's force="mesh" to survive the load. The obj_recon layout next to
    # it carries the metric scale (1 unit = 1 m in this fixture).
    mesh_dir = clip / "obj_recon" / "meshes" / "000000" / "thing"
    mesh_dir.mkdir(parents=True)
    (clip / "mesh.obj").write_text(
        "v 0 0 0\nv 0.1 0 0\nv 0 0 0.0001\nf 1 2 3\n", encoding="utf-8"
    )
    import shutil

    shutil.copy(clip / "mesh.obj", mesh_dir / "thing.obj")
    _write(
        clip / "obj_recon" / "meshes" / "000000" / "layout.json",
        {"objects": [{"local_to_scene": {"scale": [1.0, 1.0, 1.0]}}]},
    )

    _write(clip / "hand_recon" / "hands.json", {"meshes_npz": str(clip / "m.npz")})
    n = n_frames
    np.savez(
        clip / "m.npz",
        right_valid=np.ones(n, dtype=bool),
        left_valid=np.zeros(n, dtype=bool),
        right_joints=np.zeros((n, 21, 3)),
        left_joints=np.zeros((n, 21, 3)),
    )

    frames = clip / "process" / "frames"
    frames.mkdir(parents=True)
    _write(clip / "process" / "frames.json", {"frames_dir": str(frames)})

    # geometry + segment stages (needed by the mesh-scale estimate): a 100 px
    # plane at z=0.4 holding a mask whose points span 1.0 raw unit in x.
    frame_dir = clip / "geometry" / "frames" / "000000"
    frame_dir.mkdir(parents=True)
    points = np.zeros((100, 100, 3))
    points[40:60, 40:60, 0] = np.linspace(0.0, 1.0, 20)[None, :]
    points[:, :, 2] = 0.4
    np.save(frame_dir / "points.npy", points)
    np.save(
        frame_dir / "intrinsics.npy",
        np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]]),
    )
    _write(
        clip / "geometry" / "geometry.json",
        {
            "entries": [
                {"index": i, "frame_dir": str(frame_dir)} for i in range(n_frames)
            ]
        },
    )

    import cv2

    mask_dir = clip / "segment" / "masks"
    mask_dir.mkdir(parents=True)
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 40:60] = 255
    cv2.imwrite(str(mask_dir / "000000.png"), mask)
    _write(
        clip / "segment" / "masks.json",
        {
            "masks_dir": str(mask_dir),
            "entries": [
                {"index": i, "mask_filename": "000000.png", "has_mask": True}
                for i in range(n_frames)
            ],
        },
    )
    return clip


def test_load_object_trajectory(tmp_path: Path):
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    np.savetxt(poses_dir / "000000.txt", pose)
    entries = [
        {"index": 0, "tracked": True, "pose_filename": "000000.txt"},
        {"index": 1, "tracked": False, "pose_filename": None},
    ]
    trans, quat, valid = _load_object_trajectory(poses_dir, entries)
    assert valid.tolist() == [True, False]
    assert np.allclose(trans[0], [1.0, 2.0, 3.0])
    assert np.allclose(quat[1], [1.0, 0, 0, 0])


def test_load_clip_inputs_auto_embodiment(tmp_path: Path):
    clip = _make_clip(tmp_path)
    inputs = load_clip_inputs(
        clip,
        task="clip_a",
        gravity_up_cam=np.array([0.0, 0.0, 1.0]),
    )
    assert inputs.task == "clip_a"
    assert inputs.object_name == "thing"
    assert inputs.embodiment_type == "right"  # only right_valid has >=2 frames
    assert inputs.obj_valid.sum() == 4
    assert np.allclose(inputs.obj_trans_cam[-1], [0.03, 0.0, 0.5])


def test_auto_embodiment_uses_object_contact_not_visibility(tmp_path: Path):
    clip = _make_clip(tmp_path)
    n = 4
    left = np.zeros((n, 8, 3), dtype=np.float64)
    right = np.ones((n, 8, 3), dtype=np.float64) * 2.0
    for index in range(n):
        left[index, :, 0] = 0.01 * index + np.linspace(0.0, 0.08, 8)
        left[index, :, 2] = 0.5
    np.savez(
        clip / "m.npz",
        right_valid=np.ones(n, dtype=bool),
        left_valid=np.ones(n, dtype=bool),
        right_joints=np.zeros((n, 21, 3)),
        left_joints=np.zeros((n, 21, 3)),
        right_vertices=right,
        left_vertices=left,
    )

    inputs = load_clip_inputs(
        clip,
        task="clip_a",
        gravity_up_cam=np.array([0.0, 0.0, 1.0]),
    )

    assert inputs.embodiment_type == "left"


def _write_alignment(
    clip: Path,
    *,
    status: str = "accepted",
    usable: bool = True,
    indices: list[int] | None = None,
    translation_x: float = 1.0,
) -> Path:
    source = json.loads((clip / "pose_estimation" / "poses.json").read_text())
    indices = (
        indices
        if indices is not None
        else [entry["index"] for entry in source["entries"]]
    )
    poses_dir = clip / "hand_object_alignment" / "poses"
    poses_dir.mkdir(parents=True)
    entries = []
    for index in indices:
        filename = f"{index:06d}.txt"
        pose = np.eye(4)
        pose[:3, 3] = [translation_x + index, 0.25, 0.75]
        np.savetxt(poses_dir / filename, pose)
        entries.append({"index": index, "tracked": True, "pose_filename": filename})
    manifest = clip / "hand_object_alignment" / "poses.json"
    _write(
        manifest,
        {
            "schema_version": "1.0",
            "stage": "hand_object_alignment",
            "status": status,
            "usable": usable,
            "source_poses_json": str(
                (clip / "pose_estimation" / "poses.json").resolve()
            ),
            "poses_dir": str(poses_dir.resolve()),
            "validation": {"passed": status == "accepted"},
            "entries": entries,
        },
    )
    return manifest


def test_load_clip_inputs_accepts_alignment_from_filtered_poses(tmp_path: Path):
    clip = _make_clip(tmp_path)
    filtered = clip / "pose_estimation" / "poses_filtered.json"
    filtered.write_text((clip / "pose_estimation" / "poses.json").read_text())
    manifest = _write_alignment(clip)
    payload = json.loads(manifest.read_text())
    payload["source_poses_json"] = str(filtered.resolve())
    _write(manifest, payload)

    inputs = load_clip_inputs(
        clip, task="clip_a", gravity_up_cam=np.array([0.0, 0.0, 1.0])
    )
    assert np.allclose(inputs.obj_trans_cam[2], [3.0, 0.25, 0.75])


def test_load_clip_inputs_accepted_alignment_overrides_only_trajectory(tmp_path: Path):
    clip = _make_clip(tmp_path)
    _write_alignment(clip)
    inputs = load_clip_inputs(
        clip, task="clip_a", gravity_up_cam=np.array([0.0, 0.0, 1.0])
    )
    assert np.allclose(inputs.obj_trans_cam[2], [3.0, 0.25, 0.75])
    assert inputs.object_name == "thing"
    assert inputs.mesh_obj_path.name == "thing.obj"
    assert inputs.mesh_scale == pytest.approx(1.0)


def test_load_clip_inputs_canonical_selection_ignores_accepted_alignment(
    tmp_path: Path,
):
    clip = _make_clip(tmp_path)
    _write_alignment(clip)
    inputs = load_clip_inputs(
        clip,
        task="clip_a",
        gravity_up_cam=np.array([0.0, 0.0, 1.0]),
        object_trajectory="canonical",
    )
    assert np.allclose(inputs.obj_trans_cam[-1], [0.03, 0.0, 0.5])


@pytest.mark.parametrize("status", ["disabled", "rejected"])
def test_load_clip_inputs_auto_falls_back_for_explicit_non_use(
    tmp_path: Path, status: str
):
    clip = _make_clip(tmp_path)
    _write_alignment(clip, status=status, usable=False)
    inputs = load_clip_inputs(
        clip, task="clip_a", gravity_up_cam=np.array([0.0, 0.0, 1.0])
    )
    assert np.allclose(inputs.obj_trans_cam[-1], [0.03, 0.0, 0.5])


def test_load_clip_inputs_aligned_requires_accepted_manifest(tmp_path: Path):
    clip = _make_clip(tmp_path)
    with pytest.raises(FileNotFoundError, match="requested alignment manifest"):
        load_clip_inputs(
            clip,
            task="clip_a",
            gravity_up_cam=np.array([0.0, 0.0, 1.0]),
            object_trajectory="aligned",
        )


def test_load_clip_inputs_sparse_alignment_preserves_indices(tmp_path: Path):
    clip = _make_clip(tmp_path)
    _write_alignment(clip, indices=[0, 3])
    inputs = load_clip_inputs(
        clip, task="clip_a", gravity_up_cam=np.array([0.0, 0.0, 1.0])
    )
    assert inputs.obj_valid.tolist() == [True, False, False, True]
    assert np.allclose(inputs.obj_trans_cam[3], [4.0, 0.25, 0.75])


def test_load_clip_inputs_malformed_accepted_alignment_fails(tmp_path: Path):
    clip = _make_clip(tmp_path)
    manifest = _write_alignment(clip)
    payload = json.loads(manifest.read_text())
    payload["entries"].append(dict(payload["entries"][0]))
    _write(manifest, payload)
    with pytest.raises(ValueError, match="duplicate trajectory frame index"):
        load_clip_inputs(clip, task="clip_a", gravity_up_cam=np.array([0.0, 0.0, 1.0]))


def test_load_clip_inputs_requires_gravity(tmp_path: Path):
    clip = _make_clip(tmp_path)
    with pytest.raises(ValueError, match="gravity_up_cam"):
        load_clip_inputs(clip, task="clip_a")


def test_estimate_mesh_scale_match(tmp_path: Path):
    """Extent fit on `_make_clip`: the pointmap spans 1 m over a 0.1-unit mesh."""

    clip = _make_clip(tmp_path)
    trans = np.array([[0.0, 0.0, 0.4]])
    valid = np.array([True])
    scale = estimate_mesh_scale(clip, clip / "mesh.obj", trans, valid, ref_frame=0)
    # Extent ratio: 1.0 m / 0.1 unit = 10; the silhouette fit is a lower
    # bound (mask diag 28 px × z/f = 0.113 m/unit < 10), so max picks 10.
    assert scale == pytest.approx(10.0, rel=0.05)
