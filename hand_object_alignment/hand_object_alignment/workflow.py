"""Validation, manifest I/O, and orchestration for corrective alignment."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from hand_object_alignment import __version__
from hand_object_alignment.alignment import (
    FitResult,
    apply_camera_correction,
    correction_matrices,
    correction_matrix,
    fit_pose_corrections,
)


@dataclass
class AlignmentArgs:
    clip_root: Path
    """Clip output root containing hand_recon/ and pose_estimation/."""

    output_dir: Path | None = None
    """Output directory; defaults to <clip-root>/hand_object_alignment."""

    poses_json: Path | None = None
    """Object trajectory manifest. Defaults to pose_estimation/poses_filtered.json
    when present, otherwise pose_estimation/poses.json."""

    mode: Literal["auto_per_frame", "auto_global", "manual"] = "auto_per_frame"
    """Correction source: per-frame automatic fit, one global automatic fit,
    or the manual translation/rotation override below."""

    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Manual mode only: camera-frame translation correction in metres."""

    rotation_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Manual mode only: camera-frame axis-angle correction in radians."""

    enabled: bool = True
    """Disable to emit an explicit non-usable manifest without corrected poses."""

    # --- fit trust region (auto modes) -------------------------------------
    max_translation_m: float = 0.05
    """Hard clamp on |translation| of any accepted automatic correction."""

    max_rotation_deg: float = 15.0
    """Hard clamp on rotation magnitude of any accepted automatic correction."""

    translation_grid_range_m: float = 0.06
    """Warmup seed search radius around the source translation."""

    translation_grid_step_m: float = 0.01
    """Warmup seed search grid step."""

    enable_penetration_term: bool = False
    """Penalize object vertices inside a hand hull. Disabled by default because
    a merged two-hand hull includes the empty space between the hands."""

    # --- measurable acceptance gates ---------------------------------------
    min_tracked_frames: int = 2
    """Minimum corrected tracked poses required for acceptance."""

    min_inhand_overlap_frames: int = 2
    """Minimum frames with BOTH a tracked pose and a valid hand (auto modes)."""

    contact_dist_m: float = 0.02
    """Distance threshold selecting genuine hand-object contact frames and
    gating their median post-fit clearance."""

    max_penetration_m: float = 0.005
    """Gate: max post-fit penetration depth must not exceed this."""

    require_monotonic: bool = True
    """Gate: reject if the median post-fit distance is not <= the pre-fit one."""

    fail_on_rejection: bool = True
    """Raise after writing a rejected manifest when validation fails."""

    force: bool = False
    """Replace an existing alignment stage directory."""


@dataclass(frozen=True)
class AlignmentOutputs:
    stage_dir: Path
    poses_dir: Path
    poses_json_path: Path
    config_json_path: Path
    status: Literal["accepted", "rejected", "disabled"]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required manifest not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"manifest must contain a JSON object: {path}")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_entries(entries: object) -> list[dict]:
    if not isinstance(entries, list):
        raise TypeError("pose_estimation entries must be a list")
    validated: list[dict] = []
    seen: set[int] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"pose entry {position} must be an object")
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"pose entry {position} has invalid index {index!r}")
        if index in seen:
            raise ValueError(f"duplicate pose frame index: {index}")
        seen.add(index)
        tracked = entry.get("tracked")
        if not isinstance(tracked, bool):
            raise TypeError(f"pose entry {index} tracked must be boolean")
        filename = entry.get("pose_filename")
        if tracked and (not isinstance(filename, str) or not filename):
            raise ValueError(f"tracked pose entry {index} has no pose_filename")
        validated.append(dict(entry))
    return validated


def _load_rigid_pose(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"tracked pose file not found: {path}")
    try:
        pose = np.loadtxt(path, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"could not read pose matrix {path}: {exc}") from exc
    if pose.size != 16:
        raise ValueError(f"pose matrix must contain 16 values: {path}")
    pose = pose.reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError(f"pose matrix contains non-finite values: {path}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"pose matrix has invalid homogeneous row: {path}")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ValueError(f"pose matrix rotation is not rigid: {path}")
    return pose


def _manifest_base(args: AlignmentArgs, source_poses: Path, source_hands: Path) -> dict:
    payload: dict = {
        "schema_version": "1.0",
        "stage": "hand_object_alignment",
        "producer_version": __version__,
        "source_poses_json": str(source_poses),
        "source_hands_json": str(source_hands),
        "selection_contract": "trajectory_override_only",
        "pose_convention": "object_in_camera_opencv",
        "translation_units": "metres",
        "rotation_units": "radians_axis_angle",
        "mode": args.mode,
        "gating": {
            "contact_dist_m": args.contact_dist_m,
            "max_penetration_m": args.max_penetration_m,
            "min_tracked_frames": args.min_tracked_frames,
            "min_inhand_overlap_frames": args.min_inhand_overlap_frames,
            "require_monotonic": args.require_monotonic,
            "max_translation_m": args.max_translation_m,
            "max_rotation_deg": args.max_rotation_deg,
        },
    }
    if args.mode == "manual":
        payload["correction"] = {
            "composition": "corrected_object_in_camera = camera_correction @ source_object_in_camera",
            "translation_xyz": list(args.translation_xyz),
            "rotation_rotvec": list(args.rotation_rotvec),
            "global": True,
        }
    return payload


def _fit_stats_manifest(fit: FitResult) -> dict:
    frames = []
    for f in fit.frames:
        frames.append(
            {
                "frame_index": f.frame_index,
                "pre_min_dist_m": f.pre_min_dist_m,
                "pre_penetration_depth_m": f.pre_penetration_depth_m,
                "post_min_dist_m": f.post_min_dist_m,
                "post_penetration_depth_m": f.post_penetration_depth_m,
                "translation_xyz": list(f.translation_xyz),
                "rotation_rotvec": list(f.rotation_rotvec),
                "clamped": f.clamped,
                "converged": f.converged,
                "objective": f.objective,
            }
        )
    return {"per_frame": frames, "aggregate": fit.stats}


def _acceptance_gates(
    stats: dict, args: AlignmentArgs, tracked_count: int
) -> tuple[bool, list[str]]:
    """Boolean acceptance + human-readable gate failures. All measurable."""

    failures: list[str] = []
    if tracked_count < args.min_tracked_frames:
        failures.append(
            f"tracked_count {tracked_count} < min_tracked_frames {args.min_tracked_frames}"
        )
    if stats["frame_count"] < args.min_inhand_overlap_frames:
        failures.append(
            f"in-hand overlap frames {stats['frame_count']} < "
            f"min_inhand_overlap_frames {args.min_inhand_overlap_frames}"
        )
    if stats["post_min_dist_median_m"] > args.contact_dist_m:
        failures.append(
            f"median post-fit min distance {stats['post_min_dist_median_m']:.4f} m > "
            f"contact_dist_m {args.contact_dist_m:.4f} m"
        )
    if stats["post_penetration_max_m"] > args.max_penetration_m:
        failures.append(
            f"max post-fit penetration {stats['post_penetration_max_m']:.4f} m > "
            f"max_penetration_m {args.max_penetration_m:.4f} m"
        )
    if (
        args.require_monotonic
        and stats["post_min_dist_median_m"] > stats["pre_min_dist_median_m"]
    ):
        failures.append(
            f"median post-fit min distance {stats['post_min_dist_median_m']:.4f} m "
            f"worse than pre-fit {stats['pre_min_dist_median_m']:.4f} m"
        )
    if stats["translation_norm_max_m"] > args.max_translation_m + 1e-6:
        failures.append(
            f"max correction translation {stats['translation_norm_max_m']:.4f} m exceeds "
            f"trust region {args.max_translation_m:.4f} m"
        )
    if stats["rotation_deg_max"] > args.max_rotation_deg + 1e-3:
        failures.append(
            f"max correction rotation {stats['rotation_deg_max']:.2f} deg exceeds "
            f"trust region {args.max_rotation_deg:.2f} deg"
        )
    return (not failures), failures


def _collect_inhand_evidence(
    entries: list[dict],
    source_dir: Path,
    meshes: np.lib.npyio.NpzFile,
    hand_frame_count: int,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """(frame indices, source poses, hand verts) for frames with pose AND a valid hand."""

    left_valid = np.asarray(meshes["left_valid"], dtype=bool)
    right_valid = np.asarray(meshes["right_valid"], dtype=bool)
    left_verts = np.asarray(meshes["left_vertices"], dtype=np.float64)
    right_verts = np.asarray(meshes["right_vertices"], dtype=np.float64)

    frame_indices: list[int] = []
    source_poses: list[np.ndarray] = []
    hand_verts_list: list[np.ndarray] = []
    for entry in entries:
        if not entry["tracked"]:
            continue
        index = int(entry["index"])
        if index >= hand_frame_count:
            continue
        sides: list[np.ndarray] = []
        if left_valid[index]:
            sides.append(left_verts[index])
        if right_valid[index]:
            sides.append(right_verts[index])
        if not sides:
            continue
        pose = _load_rigid_pose(source_dir / entry["pose_filename"])
        frame_indices.append(index)
        source_poses.append(pose)
        hand_verts_list.append(np.concatenate(sides, axis=0))
    return frame_indices, source_poses, hand_verts_list


def _load_mesh_vertices(mesh_path: Path) -> tuple[np.ndarray, object]:
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 8:
        raise ValueError(f"object mesh has too few vertices: {mesh_path}")
    if not np.isfinite(verts).all():
        raise ValueError(f"object mesh contains non-finite vertices: {mesh_path}")
    return verts, mesh


def run_alignment(args: AlignmentArgs) -> AlignmentOutputs:
    clip_root = args.clip_root.expanduser().resolve()
    if args.poses_json is not None:
        source_poses = args.poses_json.expanduser().resolve()
    else:
        filtered = clip_root / "pose_estimation" / "poses_filtered.json"
        source_poses = (
            filtered
            if filtered.is_file()
            else clip_root / "pose_estimation" / "poses.json"
        )
    source_hands = clip_root / "hand_recon" / "hands.json"
    stage_dir = (
        (args.output_dir or clip_root / "hand_object_alignment").expanduser().resolve()
    )
    poses_dir = stage_dir / "poses"
    manifest_path = stage_dir / "poses.json"
    config_path = stage_dir / "config.json"

    if args.min_tracked_frames < 1:
        raise ValueError("--min-tracked-frames must be at least 1")
    if stage_dir.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {stage_dir}; pass --force")
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    config = asdict(args)
    config.update(
        {
            "clip_root": str(clip_root),
            "output_dir": str(stage_dir),
            "translation_xyz": list(args.translation_xyz),
            "rotation_rotvec": list(args.rotation_rotvec),
        }
    )
    _write_json(config_path, config)

    base = _manifest_base(args, source_poses, source_hands)
    if not args.enabled:
        _write_json(
            manifest_path,
            {
                **base,
                "status": "disabled",
                "usable": False,
                "poses_dir": None,
                "validation": {"passed": True, "errors": []},
                "entries": [],
            },
        )
        return AlignmentOutputs(
            stage_dir, poses_dir, manifest_path, config_path, "disabled"
        )

    def _reject(exc: Exception) -> AlignmentOutputs | None:
        shutil.rmtree(poses_dir, ignore_errors=True)
        _write_json(
            manifest_path,
            {
                **base,
                "status": "rejected",
                "usable": False,
                "poses_dir": None,
                "validation": {"passed": False, "errors": [str(exc)]},
                "entries": [],
            },
        )
        if args.fail_on_rejection:
            raise RuntimeError(f"alignment rejected: {exc}") from exc
        return AlignmentOutputs(
            stage_dir, poses_dir, manifest_path, config_path, "rejected"
        )

    try:
        poses_manifest = _read_json(source_poses)
        hands_manifest = _read_json(source_hands)
        entries = _validate_entries(poses_manifest.get("entries"))
        meshes_path = Path(hands_manifest["meshes_npz"]).expanduser().resolve()
        if not meshes_path.is_file():
            raise FileNotFoundError(f"hand meshes file not found: {meshes_path}")
        with np.load(meshes_path) as meshes:
            if not {"left_valid", "right_valid"}.issubset(meshes.files):
                raise ValueError(
                    "hand meshes must contain left_valid and right_valid arrays"
                )
            if args.mode != "manual" and not {
                "left_vertices",
                "right_vertices",
            }.issubset(meshes.files):
                raise ValueError(
                    "auto alignment modes need left_vertices/right_vertices in the hand meshes"
                )
            hand_lengths = {
                len(np.asarray(meshes[key])) for key in ("left_valid", "right_valid")
            }
            if len(hand_lengths) != 1:
                raise ValueError(
                    "hand meshes must contain equal-length left_valid and right_valid arrays"
                )
            hand_frame_count = hand_lengths.pop()
            if any(entry["index"] >= hand_frame_count for entry in entries):
                raise ValueError(
                    "pose entry index exceeds the hand trajectory frame range"
                )
            meshes_arrays = {
                key: np.asarray(meshes[key])
                for key in (
                    "left_valid",
                    "right_valid",
                    "left_vertices",
                    "right_vertices",
                )
                if key in meshes.files
            }

        source_dir_value = poses_manifest.get("poses_dir")
        source_dir = (
            Path(source_dir_value).expanduser().resolve()
            if isinstance(source_dir_value, str) and source_dir_value
            else source_poses.parent / "poses"
        )

        # Manual mode: one freely chosen global rigid correction.
        if args.mode == "manual":
            correction = correction_matrix(args.translation_xyz, args.rotation_rotvec)
            poses_dir.mkdir()
            output_entries: list[dict] = []
            tracked_count = 0
            for entry in entries:
                output = dict(entry)
                if entry["tracked"]:
                    source_pose = _load_rigid_pose(source_dir / entry["pose_filename"])
                    corrected = apply_camera_correction(source_pose, correction)
                    filename = f"{entry['index']:06d}.txt"
                    np.savetxt(poses_dir / filename, corrected)
                    output["pose_filename"] = filename
                    tracked_count += 1
                else:
                    output["pose_filename"] = None
                output_entries.append(output)
            if tracked_count < args.min_tracked_frames:
                raise ValueError(
                    f"only {tracked_count} tracked frames; require {args.min_tracked_frames}"
                )
            manifest = {
                **base,
                "status": "accepted",
                "usable": True,
                "poses_dir": str(poses_dir),
                "frame_count": hand_frame_count,
                "tracked_count": tracked_count,
                "fit_stats": None,
                "validation": {"passed": True, "errors": []},
                "entries": output_entries,
            }
            _write_json(manifest_path, manifest)
            return AlignmentOutputs(
                stage_dir, poses_dir, manifest_path, config_path, "accepted"
            )

        class _Meshes:
            def __getitem__(self, key: str) -> np.ndarray:
                return meshes_arrays[key]

        frame_indices, source_poses_list, hand_verts_list = _collect_inhand_evidence(
            entries, source_dir, _Meshes(), hand_frame_count
        )

        mesh_path_value = poses_manifest.get("mesh_path")
        if not isinstance(mesh_path_value, str) or not mesh_path_value:
            raise ValueError("pose_estimation manifest has no mesh_path")
        mesh_path = Path(mesh_path_value).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"object mesh not found: {mesh_path}")
        object_verts_local, object_mesh = _load_mesh_vertices(mesh_path)
        mesh_scale = poses_manifest.get("mesh_scale")
        if mesh_scale is None or not np.isfinite(mesh_scale) or float(mesh_scale) <= 0:
            raise ValueError("pose_estimation manifest has no valid mesh_scale")
        object_verts_local = object_verts_local * float(mesh_scale)
        object_mesh = object_mesh.copy()
        object_mesh.vertices = np.asarray(object_mesh.vertices) * float(mesh_scale)

        contact_indices: list[int] = []
        contact_poses: list[np.ndarray] = []
        contact_hands: list[np.ndarray] = []
        for index, pose, hand_verts in zip(
            frame_indices, source_poses_list, hand_verts_list
        ):
            object_cam = object_verts_local @ pose[:3, :3].T + pose[:3, 3]
            distance = float(cKDTree(hand_verts).query(object_cam, k=1)[0].min())
            if distance <= args.contact_dist_m:
                contact_indices.append(index)
                contact_poses.append(pose)
                contact_hands.append(hand_verts)
        if len(contact_indices) < args.min_inhand_overlap_frames:
            raise ValueError(
                f"only {len(contact_indices)} frames within contact_dist_m "
                f"{args.contact_dist_m:.4f}; require {args.min_inhand_overlap_frames}"
            )

        fit = fit_pose_corrections(
            source_poses=contact_poses,
            frame_indices=contact_indices,
            hand_verts_list=contact_hands,
            object_verts_local=object_verts_local,
            object_mesh=object_mesh,
            mode="global" if args.mode == "auto_global" else "per_frame",
            translation_grid_step_m=args.translation_grid_step_m,
            translation_grid_range_m=args.translation_grid_range_m,
            max_translation_m=args.max_translation_m,
            max_rotation_deg=args.max_rotation_deg,
            enable_penetration_term=args.enable_penetration_term,
        )
        corrections = {
            frame.frame_index: corr
            for frame, corr in zip(fit.frames, correction_matrices(fit))
        }

        poses_dir.mkdir()
        output_entries = []
        tracked_count = 0
        for entry in entries:
            output = dict(entry)
            if entry["tracked"]:
                source_pose = _load_rigid_pose(source_dir / entry["pose_filename"])
                corr = corrections.get(int(entry["index"]))
                corrected = (
                    apply_camera_correction(source_pose, corr)
                    if corr is not None
                    else source_pose
                )
                filename = f"{entry['index']:06d}.txt"
                np.savetxt(poses_dir / filename, corrected)
                output["pose_filename"] = filename
                output["fit_applied"] = corr is not None
                tracked_count += 1
            else:
                output["pose_filename"] = None
            output_entries.append(output)

        passed, gate_errors = _acceptance_gates(fit.stats, args, tracked_count)
        manifest = {
            **base,
            "status": "accepted" if passed else "rejected",
            "usable": passed,
            "poses_dir": str(poses_dir) if passed else None,
            "frame_count": hand_frame_count,
            "tracked_count": tracked_count,
            "inhand_overlap_frames": fit.stats["frame_count"],
            "fit_stats": _fit_stats_manifest(fit),
            "correction": {
                "composition": "corrected_object_in_camera = frame_correction @ source_object_in_camera",
                "global": args.mode == "auto_global",
            },
            "validation": {"passed": passed, "errors": gate_errors},
            "entries": output_entries if passed else [],
        }
        _write_json(manifest_path, manifest)
        if not passed:
            shutil.rmtree(poses_dir, ignore_errors=True)
            if args.fail_on_rejection:
                raise RuntimeError("alignment rejected: " + "; ".join(gate_errors))
            return AlignmentOutputs(
                stage_dir, poses_dir, manifest_path, config_path, "rejected"
            )
        return AlignmentOutputs(
            stage_dir, poses_dir, manifest_path, config_path, "accepted"
        )
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("alignment rejected:"):
            raise
        return _reject(exc) or AlignmentOutputs(
            stage_dir, poses_dir, manifest_path, config_path, "rejected"
        )
