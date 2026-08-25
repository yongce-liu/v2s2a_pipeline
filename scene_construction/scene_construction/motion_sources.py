"""Load the v2s2a pipeline's per-stage outputs as scene-construction inputs.

``process_dataset.py`` consumes trajectories in the metric, z-forward camera
frame. This module builds them from the v2s2a stage manifests:

* ``hand_recon/hands.json`` → per-frame MANO joints/vertices (camera frame,
  metres — HaWoR metric output);
* ``pose_estimation/poses.json`` → per-frame object poses (bbox-centered mesh
  frame → camera frame, metres — FoundationPose metric output);
* ``process/frames.json`` → RGB frames (for gravity estimation);
* ``geometry/geometry.json`` + ``segment/masks.json`` → metric-scale
  calibration of the SAM3D object mesh against the pose trajectory.

The do-as-i-do equivalents are ``all_hand_meshes.npz``,
``stage4_optimized/<obj>/layout_camera_frame_optimized.json`` and
``stage2_gravity/gravity.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

# The SAM3D mesh is in arbitrary (normalized) units; poses come from a
# different estimator whose depth magnitude inherits the same mesh scale. The
# depth-matching estimate derives the raw-units → metres factor (the
# ``scale_m`` of do-as-i-do's track_object_foundationpose.py).
MIN_MASK_PIXELS = 50


@dataclass(frozen=True)
class ClipInputs:
    """Everything ``process_dataset`` needs for one clip, camera frame, metres."""

    task: str
    object_name: str
    frames_dir: Path
    mesh_obj_path: Path
    hand_npz_path: Path
    embodiment_type: str
    obj_trans_cam: np.ndarray  # (N, 3), bbox-center translation, metres
    obj_quat_cam: np.ndarray  # (N, 4) wxyz, camera frame
    obj_valid: np.ndarray  # (N,) bool
    mesh_scale: float  # raw mesh units → metres (same convention as pose)
    gravity_up_cam: np.ndarray  # (3,) world-up unit vector in camera frame
    gravity_meta: dict


def _load_json(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required stage manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_dir(clip_root: Path, stage: str) -> Path:
    stage_dir = clip_root / stage
    if not stage_dir.is_dir():
        raise FileNotFoundError(
            f"Stage output missing: {stage_dir} (run `{stage}` first)"
        )
    return stage_dir


def _resolve_hand_npz(clip_root: Path) -> Path:
    hands = _load_json(_stage_dir(clip_root, "hand_recon") / "hands.json")
    npz_path = Path(hands["meshes_npz"])
    if not npz_path.exists():
        raise FileNotFoundError(f"hand_recon meshes.npz not found: {npz_path}")
    return npz_path


def _resolve_frames_dir(clip_root: Path) -> Path:
    frames_json = _load_json(_stage_dir(clip_root, "process") / "frames.json")
    frames_dir = Path(frames_json["frames_dir"])
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"process frames dir not found: {frames_dir}")
    return frames_dir


def _validate_trajectory_entries(entries: object, source: Path) -> list[dict]:
    if not isinstance(entries, list):
        raise TypeError(f"trajectory entries must be a list: {source}")
    validated: list[dict] = []
    seen: set[int] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"trajectory entry {position} must be an object: {source}")
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(
                f"trajectory entry {position} has invalid index {index!r}: {source}"
            )
        if index in seen:
            raise ValueError(f"duplicate trajectory frame index {index}: {source}")
        seen.add(index)
        if not isinstance(entry.get("tracked"), bool):
            raise TypeError(
                f"trajectory entry {index} tracked must be boolean: {source}"
            )
        pose_filename = entry.get("pose_filename")
        if entry["tracked"] and (
            not isinstance(pose_filename, str) or not pose_filename
        ):
            raise ValueError(
                f"tracked trajectory entry {index} has no pose_filename: {source}"
            )
        validated.append(entry)
    return validated


def _load_object_trajectory(
    poses_dir: Path, entries: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame bbox-center translation + wxyz quaternion from pose files."""

    entries = _validate_trajectory_entries(entries, poses_dir)
    by_index = {entry["index"]: entry for entry in entries}
    indices = sorted(by_index)
    n = indices[-1] + 1 if indices else 0
    trans = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    quat[:, 0] = 1.0
    valid = np.zeros(n, dtype=bool)
    for idx in indices:
        entry = by_index[idx]
        pose_filename = entry.get("pose_filename")
        if not entry["tracked"]:
            continue
        pose_path = poses_dir / pose_filename
        if not pose_path.is_file():
            raise FileNotFoundError(f"tracked object pose not found: {pose_path}")
        try:
            pose_values = np.loadtxt(pose_path, dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"could not read object pose {pose_path}: {exc}") from exc
        if pose_values.size != 16:
            raise ValueError(f"object pose must contain 16 values: {pose_path}")
        pose = pose_values.reshape(4, 4)
        if not np.isfinite(pose).all():
            raise ValueError(f"object pose contains non-finite values: {pose_path}")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(f"object pose has invalid homogeneous row: {pose_path}")
        rotation = pose[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-4
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"object pose rotation is not rigid: {pose_path}")
        rot = Rotation.from_matrix(rotation).as_quat()  # xyzw
        trans[idx] = pose[:3, 3]
        quat[idx] = rot[[3, 0, 1, 2]]
        valid[idx] = True
    return trans, quat, valid


def _resolve_object_trajectory(
    clip_root: Path,
    canonical_poses_dir: Path,
    canonical_manifest: dict,
    selection: Literal["auto", "canonical", "aligned"],
    alignment_manifest_path: Path | None,
) -> tuple[Path, list[dict]]:
    """Select an optional validated override without changing canonical metadata."""

    if selection == "canonical":
        return canonical_poses_dir, canonical_manifest.get("entries", [])

    manifest_path = (
        alignment_manifest_path.expanduser().resolve()
        if alignment_manifest_path is not None
        else clip_root / "hand_object_alignment" / "poses.json"
    )
    if not manifest_path.exists():
        if selection == "aligned":
            raise FileNotFoundError(
                f"requested alignment manifest not found: {manifest_path}"
            )
        return canonical_poses_dir, canonical_manifest.get("entries", [])

    alignment = _load_json(manifest_path)
    if alignment.get("stage") != "hand_object_alignment":
        raise ValueError(f"invalid alignment stage in {manifest_path}")
    status = alignment.get("status")
    usable = alignment.get("usable")
    if status in {"disabled", "rejected"} and usable is False:
        if selection == "aligned":
            raise ValueError(f"requested alignment is {status}: {manifest_path}")
        return canonical_poses_dir, canonical_manifest.get("entries", [])
    if status != "accepted" or usable is not True:
        raise ValueError(
            f"alignment manifest has inconsistent status/usable fields: {manifest_path}"
        )
    if alignment.get("validation", {}).get("passed") is not True:
        raise ValueError(f"accepted alignment failed validation: {manifest_path}")
    source_poses = alignment.get("source_poses_json")
    allowed_sources = {
        (canonical_poses_dir.parent / "poses.json").resolve(),
        (canonical_poses_dir.parent / "poses_filtered.json").resolve(),
    }
    if (
        not isinstance(source_poses, str)
        or Path(source_poses).expanduser().resolve() not in allowed_sources
    ):
        raise ValueError(
            f"alignment source_poses_json does not match this clip: {manifest_path}"
        )
    poses_dir_value = alignment.get("poses_dir")
    if not isinstance(poses_dir_value, str) or not poses_dir_value:
        raise ValueError(f"accepted alignment has no poses_dir: {manifest_path}")
    poses_dir = Path(poses_dir_value).expanduser().resolve()
    if not poses_dir.is_dir():
        raise FileNotFoundError(
            f"accepted alignment poses directory not found: {poses_dir}"
        )
    entries = _validate_trajectory_entries(alignment.get("entries"), manifest_path)
    return poses_dir, entries


def _map_entries(entries: list[dict]) -> dict[int, dict]:
    return {int(entry["index"]): entry for entry in entries}


def _load_object_mask(
    masks_manifest: dict, entry: dict, shape_hw: tuple[int, int]
) -> np.ndarray | None:
    """Object mask for a frame, resized to the pointmap resolution."""

    import cv2

    filename = entry.get("mask_filename")
    if not filename or not entry.get("has_mask"):
        return None
    path = Path(masks_manifest["masks_dir"]) / filename
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = mask > 127
    if mask.shape != shape_hw:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (shape_hw[1], shape_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def _find_pose_frame_dirs(geometry_manifest: dict) -> dict[int, Path]:
    return {
        int(entry["index"]): Path(entry["frame_dir"])
        for entry in geometry_manifest.get("entries", [])
    }


def _mesh_extent_metres(mesh_obj_path: Path, mesh_scale: float) -> np.ndarray:
    import trimesh

    verts = np.asarray(
        trimesh.load(str(mesh_obj_path), force="mesh").vertices, dtype=np.float64
    )
    return (verts.max(0) - verts.min(0)) * mesh_scale


def estimate_mesh_scale(
    clip_root: Path,
    mesh_obj_path: Path,
    obj_trans_cam: np.ndarray,
    obj_valid: np.ndarray,
    ref_frame: int,
) -> float:
    """Sanity-check the metric mesh scale by depth/mask matching at ``ref_frame``.

    With the metric-scale mesh (pose_estimation now scales the SAM3D mesh by
    the obj_recon layout before tracking), this ratio should be ≈ 1. The
    extent estimator is retained for cross-checking only — do-as-i-do relied
    on it as the primary scale estimate, but it degenerates on thin/elongated
    objects (mask pixels don't cover the full object extent, so extent ratios
    over-estimate)."""

    geometry_manifest = _load_json(_stage_dir(clip_root, "geometry") / "geometry.json")
    masks_manifest = _load_json(_stage_dir(clip_root, "segment") / "masks.json")
    frame_dirs = _find_pose_frame_dirs(geometry_manifest)
    if ref_frame not in frame_dirs:
        raise ValueError(f"ref_frame {ref_frame} missing from geometry.json")

    frame_dir = frame_dirs[ref_frame]
    points = np.load(frame_dir / "points.npy").astype(np.float64)
    # MoGe v3 writes invalid support as NaN; support pixels outside the object
    # mask are expected to stay finite. Only the masked pixels enter the fit.
    intr = np.load(frame_dir / "intrinsics.npy").astype(np.float64)
    h, w = points.shape[:2]

    mask_entry = _map_entries(masks_manifest.get("entries", [])).get(ref_frame)
    if mask_entry is None:
        raise ValueError(f"ref_frame {ref_frame} missing from masks.json")
    mask = _load_object_mask(masks_manifest, mask_entry, (h, w))
    if mask is None or int(mask.sum()) < MIN_MASK_PIXELS:
        raise ValueError(f"ref_frame {ref_frame}: object mask missing or too small")

    ys, xs = np.where(mask)
    pm = points[ys, xs]
    finite = np.isfinite(pm).all(axis=1)
    pm = pm[finite]
    if len(pm) < MIN_MASK_PIXELS:
        raise ValueError(
            f"ref_frame {ref_frame}: too few finite pointmap depths in mask"
        )

    import trimesh

    mesh = trimesh.load(str(mesh_obj_path), force="mesh")
    verts_raw = np.asarray(mesh.vertices, dtype=np.float64)
    ext_mesh = verts_raw.max(0) - verts_raw.min(0)
    ext_pm = pm.max(0) - pm.min(0)
    s_extent = float(np.max(ext_pm) / max(np.max(ext_mesh), 1e-9))

    fx, fy = intr[0, 0], intr[1, 1]
    z_med = float(np.median(pm[:, 2]))
    mask_diag = float(
        np.hypot(max(xs.max() - xs.min(), 1), max(ys.max() - ys.min(), 1))
    )
    proj_diag_per_unit = np.hypot(fx * ext_mesh.max(), fy * ext_mesh.max()) / z_med
    s_sil = mask_diag / max(proj_diag_per_unit, 1e-9)

    # Both are lower bounds (silhouette foreshortening, visible-surface extent);
    # take the max, matching do-as-i-do.
    scale_m = max(s_extent, s_sil)
    return scale_m


def load_clip_inputs(
    clip_root: Path,
    task: str,
    object_name: str | None = None,
    embodiment_type: str = "auto",
    mesh_frame: int | None = None,
    gravity_up_cam: np.ndarray | None = None,
    gravity_meta: dict | None = None,
    object_trajectory: Literal["auto", "canonical", "aligned"] = "auto",
    alignment_manifest: Path | None = None,
) -> ClipInputs:
    """Read the v2s2a stage manifests and return process_dataset's inputs."""

    clip_root = clip_root.expanduser().resolve()

    poses_dir = _stage_dir(clip_root, "pose_estimation") / "poses"
    if not poses_dir.is_dir():
        raise FileNotFoundError(f"pose_estimation poses not found: {poses_dir}")
    poses_manifest = _load_json(poses_dir.parent / "poses.json")
    mesh_obj_path = Path(poses_manifest["mesh_path"]).expanduser().resolve()
    if not mesh_obj_path.exists():
        raise FileNotFoundError(f"tracked object mesh not found: {mesh_obj_path}")
    if object_name is None:
        object_name = poses_manifest.get("object_name", mesh_obj_path.stem)

    selected_poses_dir, selected_entries = _resolve_object_trajectory(
        clip_root,
        poses_dir,
        poses_manifest,
        object_trajectory,
        alignment_manifest,
    )
    trans, quat, valid = _load_object_trajectory(selected_poses_dir, selected_entries)
    n_obj_valid = int(valid.sum())
    if n_obj_valid < 2:
        raise ValueError(
            f"object trajectory unusable: {n_obj_valid}/{len(valid)} frames tracked"
        )

    # The pose trajectory is tracked on a metric mesh (pose_estimation scales
    # the raw SAM3D mesh by the obj_recon layout scale before tracking). The
    # mesh written into the outputs must match that frame: apply the same
    # layout scale here. Translations are unaffected (the pose translation is
    # the bbox-center in metres, and scaling does not move the bbox center of
    # an origin-symmetric normalization — verified numerically against the
    # tracked trajectory).
    layout_path = next(
        (
            candidate
            for candidate in (
                mesh_obj_path.parent / "layout.json",
                mesh_obj_path.parent.parent / "layout.json",
            )
            if candidate.exists()
        ),
        None,
    )
    if layout_path is None:
        raise FileNotFoundError(
            f"obj_recon layout.json missing next to {mesh_obj_path} — cannot "
            "determine the metric mesh scale (pose_estimation applies it "
            "before tracking, and the exported mesh must match that frame)."
        )
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    mesh_scale = float(np.mean(layout["objects"][0]["local_to_scene"]["scale"]))

    # Cross-check: SAM3D's own metric scale fits the metric mesh to ~0.1–0.2 m
    # for a hand-manipulable object. Deeper extent/depth cross-checks belong
    # in a validator (the aggregate mask in masks.json covers every prompt,
    # so a spatial extent comparison against it is not meaningful).
    mesh_extent = _mesh_extent_metres(mesh_obj_path, mesh_scale)
    from loguru import logger

    logger.info(
        "[motion_sources] mesh_scale={:.4f} m/unit from {} → extent {} m",
        mesh_scale,
        layout_path,
        np.round(mesh_extent, 3).tolist(),
    )
    if not 0.01 <= float(np.max(mesh_extent)) <= 0.5:
        raise ValueError(
            f"Metric mesh extent {np.max(mesh_extent):.3f} m is implausible for a "
            "hand-manipulable object — check the obj_recon layout scale and the "
            "pose_estimation run."
        )

    hand_npz_path = _resolve_hand_npz(clip_root)
    frames_dir = _resolve_frames_dir(clip_root)

    if embodiment_type == "auto":
        with np.load(hand_npz_path) as meshes:
            has_right = int(np.asarray(meshes["right_valid"], dtype=bool).sum()) >= 2
            has_left = int(np.asarray(meshes["left_valid"], dtype=bool).sum()) >= 2
            if {"right_vertices", "left_vertices"}.issubset(meshes.files):
                import trimesh

                object_mesh = trimesh.load(str(mesh_obj_path), force="mesh")
                object_vertices = np.asarray(object_mesh.vertices, dtype=np.float64)
                object_vertices *= float(mesh_scale)
                if len(object_vertices) > 512:
                    indices = np.linspace(0, len(object_vertices) - 1, 512).astype(int)
                    object_vertices = object_vertices[indices]

                contact_counts = {}
                for side in ("right", "left"):
                    hand_vertices = np.asarray(
                        meshes[f"{side}_vertices"], dtype=np.float64
                    )
                    hand_valid = np.asarray(meshes[f"{side}_valid"], dtype=bool)
                    count = 0
                    for index in np.flatnonzero(hand_valid & valid):
                        object_world = (
                            object_vertices
                            @ Rotation.from_quat(quat[index, [1, 2, 3, 0]])
                            .as_matrix()
                            .T
                            + trans[index]
                        )
                        hand = hand_vertices[index]
                        distance = np.linalg.norm(
                            hand[:, None, :] - object_world[None, :, :], axis=2
                        ).min()
                        count += distance < 0.1
                    contact_counts[side] = count
                has_right = contact_counts["right"] >= 2
                has_left = contact_counts["left"] >= 2
                logger.info(
                    "[motion_sources] contact frames: left={} right={}",
                    contact_counts["left"],
                    contact_counts["right"],
                )
        embodiment_type = (
            "bimanual"
            if (has_right and has_left)
            else ("right" if has_right else "left")
        )
    if gravity_up_cam is None:
        raise ValueError(
            "gravity_up_cam is required: pass --gravity-json, or let the gravity "
            "stage run first (scene_construction.gravity)"
        )

    return ClipInputs(
        task=task,
        object_name=object_name,
        frames_dir=frames_dir,
        mesh_obj_path=mesh_obj_path,
        hand_npz_path=hand_npz_path,
        embodiment_type=embodiment_type,
        obj_trans_cam=trans,
        obj_quat_cam=quat,
        obj_valid=valid,
        mesh_scale=float(mesh_scale),
        gravity_up_cam=np.asarray(gravity_up_cam, dtype=np.float64),
        gravity_meta=gravity_meta or {},
    )
