import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from hand_object_alignment.workflow import AlignmentArgs, run_alignment


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _hand_ring(center: np.ndarray, radius: float = 0.06, n: int = 48) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=1
    )
    return circle + np.asarray(center, dtype=np.float64)


def _clip(tmp_path: Path, duplicate: bool = False, with_hands: bool = False) -> Path:
    """A minimal clip: 3 tracked poses; optionally a hand ring at each pose."""

    clip = tmp_path / "clip"
    poses = clip / "pose_estimation" / "poses"
    poses.mkdir(parents=True)
    entries = []
    hand_verts = np.zeros((3, 48, 3))
    for index in range(3):
        translation = np.array([index * 0.0, 0.0, 0.30])
        pose = np.eye(4)
        pose[:3, 3] = translation
        np.savetxt(poses / f"{index:06d}.txt", pose)
        entries.append(
            {"index": index, "tracked": True, "pose_filename": f"{index:06d}.txt"}
        )
        hand_verts[index] = _hand_ring(translation)
    if duplicate:
        entries[-1]["index"] = 1

    mesh_path = clip / "mesh.obj"
    trimesh.creation.icosphere(subdivisions=1, radius=0.04).export(str(mesh_path))
    _write_json(
        clip / "pose_estimation" / "poses.json",
        {"entries": entries, "mesh_path": str(mesh_path), "mesh_scale": 1.0},
    )
    np.savez(
        clip / "meshes.npz",
        left_valid=np.ones(3, bool),
        right_valid=np.zeros(3, bool),
        left_vertices=hand_verts,
        right_vertices=np.zeros_like(hand_verts),
    )
    _write_json(
        clip / "hand_recon" / "hands.json", {"meshes_npz": str(clip / "meshes.npz")}
    )
    return clip


def test_manual_mode_preserves_sources_and_writes_accepted_override(tmp_path: Path):
    clip = _clip(tmp_path)
    original = (clip / "pose_estimation" / "poses" / "000000.txt").read_text()
    outputs = run_alignment(
        AlignmentArgs(clip_root=clip, mode="manual", translation_xyz=(0.1, 0.0, 0.0))
    )
    manifest = json.loads(outputs.poses_json_path.read_text())
    corrected = np.loadtxt(outputs.poses_dir / "000000.txt")
    assert manifest["status"] == "accepted" and manifest["usable"] is True
    assert manifest["mode"] == "manual"
    assert manifest["tracked_count"] == 3
    assert manifest["fit_stats"] is None
    assert np.allclose(corrected[:3, 3], [0.1, 0.0, 0.30])
    assert (clip / "pose_estimation" / "poses" / "000000.txt").read_text() == original


def test_default_prefers_filtered_pose_manifest(tmp_path: Path):
    clip = _clip(tmp_path, with_hands=True)
    source = json.loads((clip / "pose_estimation" / "poses.json").read_text())
    source["stage"] = "pose_estimation.temporal_filter"
    source["poses_dir"] = str((clip / "pose_estimation" / "poses").resolve())
    _write_json(clip / "pose_estimation" / "poses_filtered.json", source)

    outputs = run_alignment(AlignmentArgs(clip_root=clip, mode="auto_per_frame"))
    manifest = json.loads(outputs.poses_json_path.read_text())
    assert manifest["source_poses_json"].endswith("poses_filtered.json")


def test_auto_per_frame_accepts_a_clean_grip(tmp_path: Path):
    clip = _clip(tmp_path, with_hands=True)
    outputs = run_alignment(AlignmentArgs(clip_root=clip, mode="auto_per_frame"))
    manifest = json.loads(outputs.poses_json_path.read_text())
    assert manifest["status"] == "accepted", manifest["validation"]["errors"]
    assert manifest["usable"] is True
    assert manifest["inhand_overlap_frames"] == 3
    stats = manifest["fit_stats"]["aggregate"]
    # Already-gripping hands: corrections must be small and non-destructive.
    assert stats["translation_norm_max_m"] <= 0.05
    assert stats["post_min_dist_median_m"] <= stats["pre_min_dist_median_m"] + 1e-9
    # Per-frame corrections recorded for audit.
    for frame in manifest["fit_stats"]["per_frame"]:
        assert "translation_xyz" in frame and "rotation_rotvec" in frame
        assert "post_min_dist_m" in frame


def test_auto_applies_fits_only_to_contact_frame_indices(tmp_path: Path):
    clip = _clip(tmp_path, with_hands=True)
    data = np.load(clip / "meshes.npz")
    left = data["left_vertices"].copy()
    left[1] += np.array([1.0, 0.0, 0.0])
    np.savez(
        clip / "meshes.npz",
        left_valid=data["left_valid"],
        right_valid=data["right_valid"],
        left_vertices=left,
        right_vertices=data["right_vertices"],
    )

    outputs = run_alignment(AlignmentArgs(clip_root=clip, mode="auto_per_frame"))
    manifest = json.loads(outputs.poses_json_path.read_text())
    applied = [entry["index"] for entry in manifest["entries"] if entry["fit_applied"]]
    assert applied == [0, 2]
    source_middle = np.loadtxt(clip / "pose_estimation" / "poses" / "000001.txt")
    output_middle = np.loadtxt(outputs.poses_dir / "000001.txt")
    np.testing.assert_allclose(output_middle, source_middle)


def test_auto_rejects_when_no_hands_overlap(tmp_path: Path):
    clip = _clip(tmp_path)
    # Invalidate all hands: no evidence frames remain.
    import numpy as np

    data = np.load(clip / "meshes.npz")
    np.savez(
        clip / "meshes.npz",
        left_valid=np.zeros(3, bool),
        right_valid=np.zeros(3, bool),
        left_vertices=data["left_vertices"],
        right_vertices=data["right_vertices"],
    )
    outputs = run_alignment(
        AlignmentArgs(
            clip_root=clip,
            mode="auto_per_frame",
            fail_on_rejection=False,
            min_inhand_overlap_frames=2,
        )
    )
    manifest = json.loads(outputs.poses_json_path.read_text())
    assert manifest["status"] == "rejected"
    assert manifest["usable"] is False
    # No evidence → fit fails before any pose is written.
    assert outputs.status == "rejected"


def test_auto_rejects_when_contact_gate_fails(tmp_path: Path):
    clip = _clip(tmp_path)
    # Impossible contact gate: even a correct grip cannot reach 1 mm.
    outputs = run_alignment(
        AlignmentArgs(
            clip_root=clip,
            mode="auto_per_frame",
            fail_on_rejection=False,
            contact_dist_m=1e-9,
        )
    )
    manifest = json.loads(outputs.poses_json_path.read_text())
    assert manifest["status"] == "rejected"
    assert manifest["usable"] is False
    assert manifest["poses_dir"] is None
    assert any("contact_dist_m" in e for e in manifest["validation"]["errors"])
    assert "fit_stats" not in manifest


def test_rejected_manifest_is_written_before_failure(tmp_path: Path):
    clip = _clip(tmp_path, duplicate=True)
    with pytest.raises(RuntimeError, match="duplicate pose frame index"):
        run_alignment(AlignmentArgs(clip_root=clip))
    manifest = json.loads((clip / "hand_object_alignment" / "poses.json").read_text())
    assert manifest["status"] == "rejected"
    assert manifest["validation"]["passed"] is False


def test_disabled_manifest_is_explicitly_non_usable(tmp_path: Path):
    outputs = run_alignment(AlignmentArgs(clip_root=_clip(tmp_path), enabled=False))
    manifest = json.loads(outputs.poses_json_path.read_text())
    assert manifest["status"] == "disabled"
    assert manifest["usable"] is False
