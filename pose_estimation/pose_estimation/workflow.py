"""End-to-end FoundationPose object pose estimation for one clip.

Reads the ``process`` stage's frame manifest (``frames.json``), the
``geometry`` stage's per-frame depth/points/intrinsics, and the ``segment``
stage's per-frame object masks, runs FoundationPose register-then-track over
the frames, and writes the agreed pipeline artifacts under
``<clip_root>/pose_estimation/``:

.. code-block:: text

    <clip_root>/pose_estimation/
    ├── config.json          # effective run config (same style as other stages)
    ├── poses.json           # per-frame manifest (pose paths / status)
    ├── poses/
    │   ├── 000000.txt       # 4x4 ob_in_cam matrix
    │   └── ...
    └── vis/
        ├── 000000.png       # pose overlay (inflated projected bbox)
        └── ...
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

from pose_estimation import __version__
from pose_estimation.foundationpose_estimator import (
    FoundationPoseArgs,
    FoundationPoseEstimator,
)
from pose_estimation.frames import FrameManifest, load_frame_manifest
from pose_estimation.media import (
    PoseVideoWriter,
    load_intrinsics,
    load_mask,
    load_rgb_image,
    render_mesh_overlay,
)
from pose_estimation.temporal_filter import (
    TemporalFilterArgs,
    apply_temporal_filter_to_run,
)

POSE_FILENAME_PATTERN = "{:06d}.txt"
POINTS_FILENAME = "points.npy"
INTRINSICS_FILENAME = "intrinsics.npy"
MIN_MASK_AREA_PX = 100


@dataclass
class PoseEstimationVideoArgs:
    """Arguments for pose estimation + tracking of one object over a video."""

    frames_json: Path
    """Path to the ``process`` stage's ``frames.json`` (the frame manifest)."""

    mesh_path: Path
    """Path to the object's textured mesh (``.obj``) to track."""

    masks_json: Path
    """Path to the ``segment`` stage's ``masks.json``."""

    geometry_json: Path | None = None
    """Path to the ``geometry`` stage's ``geometry.json``; omit to derive it
    from the clip root (``<clip>/geometry/geometry.json``)."""

    init_frame: int = 0
    """Reference frame index used in single-anchor mode. Frames before it are
    back-propagated from the registered pose."""

    anchor_frames: list[int] | None = None
    """Registration anchors for a non-MV mesh. An MV mesh always uses the middle
    obj_recon view and its exported pose as the single bidirectional tracking seed."""

    prompt_id: str | None = None
    """Object prompt whose per-prompt segmentation mask is used for
    registration. Defaults to the object name, with underscores replaced by
    spaces. This avoids registering against the aggregate hand+object mask."""

    object_name: str | None = None
    """Object identifier recorded in the output manifest (defaults to the
    mesh filename stem)."""

    output_root: Path | None = None
    """Root under which ``<clip_stem>/pose_estimation/`` is created. Defaults
    to the pipeline outputs root derived from ``frames_json``
    (``<clip_root>/..``), so outputs land alongside the other stages under
    ``outputs/<clip>/`` rather than inside this package."""

    reinit_every: int = 0
    """Re-run the full registration every N frames (0 = register once)."""

    translation_prior: bool = True
    """Before each tracking refinement, translate the previous pose by the
    object point-map centroid motion. This keeps fast motion inside the
    FoundationPose crop without changing its tracked rotation."""

    max_translation_step_m: float = 0.15
    """Maximum accepted object-centroid motion between adjacent frames."""

    anchor_translation_to_geometry: bool = True
    """After each FoundationPose rotation refinement, place the transformed
    metric mesh centroid at the object point-map median. This follows the
    hand/point-map translation calibration used by do-as-i-do and prevents
    accumulated depth drift."""

    mesh_scale: float | None = None
    """Uniform raw-units → metres scale applied to the mesh before tracking.

    FoundationPose assumes the mesh is already in metric units; the SAM3D
    ``.obj`` is in normalized units, so without scaling, poses come out
    centimetre-correct in depth but ~6× too large in cross-view extent (the
    refiner never fits lateral scale). ``None`` reads the scale from the
    obj_recon ``layout.json`` next to the mesh (``local_to_scene.scale``, the
    SAM3D metric fit); pass a value to override."""

    start_from_layout: bool = True
    """Anchor the metric scale to the obj_recon layout when --mesh-scale is
    not given. Disable to track in raw units."""

    max_frames: int | None = None
    """Limit the number of frames processed (None = all frames)."""

    temporal_filter: TemporalFilterArgs = field(default_factory=TemporalFilterArgs)
    """Optional late-frame error-state EKF with innovation gating. Enable with
    ``--temporal-filter.enabled``; filtered poses are written separately."""

    vis: bool = False
    """Render the tracked mesh onto each frame and write a single overlay video
    to ``<clip>/pose_estimation/vis.mp4`` (no per-frame PNGs)."""

    foundationpose: FoundationPoseArgs = field(default_factory=FoundationPoseArgs)
    """FoundationPose model settings (weights, device, refine iters, ...)."""


@dataclass(frozen=True)
class PoseEntry:
    """Per-frame pose record written into ``poses.json``."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    pose_filename: str | None
    tracked: bool
    method: str
    """``register`` / ``obj-recon-seed-refine`` / ``track`` / ``track-backward``."""
    direction: str = ""
    """``forward`` / ``backward`` / ``register`` / ``skipped``."""
    anchor_frame: int | None = None
    """Registration anchor that seeded this pose, when known."""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frame_filename": self.frame_filename,
            "timestamp_sec": self.timestamp_sec,
            "pose_filename": self.pose_filename,
            "tracked": self.tracked,
            "method": self.method,
            "direction": self.direction,
            "anchor_frame": self.anchor_frame,
        }


@dataclass(frozen=True)
class MVAnchorPose:
    """One obj_recon-provided metric object pose for an MV keyframe."""

    frame_index: int
    object_to_camera: np.ndarray
    scale: float
    reference: bool


@dataclass(frozen=True)
class PoseEstimationOutputs:
    """Everything produced by one pose-estimation run."""

    clip_root: Path
    stage_dir: Path
    poses_dir: Path
    vis_video_path: Path | None
    poses_json_path: Path
    config_json_path: Path
    entries: list[PoseEntry]
    filtered_poses_json_path: Path | None = None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_geometry_manifest(geometry_json: Path) -> dict:
    geometry_json = geometry_json.expanduser().resolve()
    if not geometry_json.exists():
        raise FileNotFoundError(f"geometry.json not found: {geometry_json}")
    return json.loads(geometry_json.read_text(encoding="utf-8"))


def load_masks_manifest(masks_json: Path) -> dict:
    masks_json = masks_json.expanduser().resolve()
    if not masks_json.exists():
        raise FileNotFoundError(f"masks.json not found: {masks_json}")
    return json.loads(masks_json.read_text(encoding="utf-8"))


def _geometry_frame_dir(geometry_manifest: dict, index: int) -> Path | None:
    for entry in geometry_manifest.get("entries", []):
        if int(entry["index"]) == index:
            return Path(entry["frame_dir"])
    return None


def _mask_path(
    masks_manifest: dict,
    index: int,
    prompt_id: str | None = None,
) -> Path | None:
    masks_dir = Path(masks_manifest.get("masks_dir", ""))
    for entry in masks_manifest.get("entries", []):
        if int(entry["index"]) != index:
            continue
        mask_entry = entry
        if prompt_id is not None:
            mask_entry = next(
                (
                    prompt_mask
                    for prompt_mask in entry.get("prompt_masks", [])
                    if prompt_mask.get("prompt_id") == prompt_id
                ),
                None,
            )
            if mask_entry is None:
                return None
        filename = mask_entry.get("mask_filename")
        if filename is None or not mask_entry.get("has_mask"):
            return None
        candidate = masks_dir / filename
        return candidate if candidate.exists() else None
    return None


def _find_layout_path(mesh_path: Path) -> Path | None:
    """Find either the MV per-object layout or legacy per-frame layout."""

    for candidate in (
        mesh_path.parent / "layout.json",
        mesh_path.parent.parent / "layout.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _is_mv_mesh(mesh_path: Path) -> bool:
    return mesh_path.expanduser().resolve().parent.parent.name == "mv"


def _load_mv_anchor_poses(mesh_path: Path) -> list[MVAnchorPose]:
    """Load the mandatory obj_recon v2 per-view metric pose contract."""

    if not _is_mv_mesh(mesh_path):
        return []
    view_poses_path = mesh_path.parent / "view_poses.json"
    if not view_poses_path.exists():
        raise FileNotFoundError(f"MV pose contract not found: {view_poses_path}")
    payload = json.loads(view_poses_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "2.0",
        "stage": "obj_recon",
        "coordinate_frame": "pytorch3d_camera",
        "pose_convention": "object_local_to_camera",
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Invalid MV pose contract {key}: expected {expected!r}, "
                f"got {payload.get(key)!r}"
            )
    views = payload.get("views", [])
    if not views:
        raise ValueError(f"No per-view poses in {view_poses_path}")

    anchors = []
    for view in views:
        matrix = np.asarray(view.get("object_to_camera_opencv"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(
                f"Invalid object_to_camera_opencv for view {view.get('view')}"
            )
        scale_xyz = np.asarray(view.get("scale"), dtype=np.float64).reshape(-1)
        if (
            scale_xyz.size != 3
            or not np.isfinite(scale_xyz).all()
            or np.any(scale_xyz <= 0)
        ):
            raise ValueError(f"Invalid metric scale for view {view.get('view')}")
        if not np.allclose(scale_xyz, scale_xyz.mean(), rtol=0.01, atol=1e-6):
            raise ValueError(
                f"MV pose scale must be uniform for view {view.get('view')}"
            )
        anchors.append(
            MVAnchorPose(
                frame_index=int(view["frame_index"]),
                object_to_camera=matrix,
                scale=float(np.mean(scale_xyz)),
                reference=bool(view.get("reference")),
            )
        )
    if len({anchor.frame_index for anchor in anchors}) != len(anchors):
        raise ValueError(f"Duplicate frame indices in {view_poses_path}")
    if sum(anchor.reference for anchor in anchors) != 1:
        raise ValueError(f"Expected exactly one reference view in {view_poses_path}")
    return anchors


def _resolve_anchor_frames(mesh_path: Path, requested: list[int] | None) -> list[int]:
    """Resolve the single MV seed or explicit non-MV registration anchors."""

    mv_anchors = _load_mv_anchor_poses(mesh_path)
    if mv_anchors:
        ordered = sorted(mv_anchors, key=lambda anchor: anchor.frame_index)
        return [ordered[(len(ordered) - 1) // 2].frame_index]
    if requested is not None:
        if not requested:
            raise ValueError("--anchor-frames must contain at least one frame.")
        return sorted(set(requested))
    return []


def _depth_from_geometry(frame_dir: Path) -> np.ndarray:
    """Metric depth from the geometry stage's point map Z channel.

    FoundationPose wants a metric depth map (metres, 0 = invalid). The
    ``geometry`` stage stores camera-space points; the Z channel is the depth,
    and invalid (inf) pixels are zeroed.
    """

    points = np.load(frame_dir / POINTS_FILENAME)
    depth = np.asarray(points[..., 2], dtype=np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.001] = 0.0
    return depth


def _skipped_entry(frame) -> PoseEntry:
    return PoseEntry(
        index=frame.index,
        frame_filename=frame.frame_filename,
        timestamp_sec=frame.timestamp_sec,
        pose_filename=None,
        tracked=False,
        method="skipped",
        direction="skipped",
    )


def _load_registration_mask(
    mask_path: Path,
    depth_shape: tuple[int, int],
) -> np.ndarray | None:
    """Load a registration mask, resized to the depth resolution.

    Returns ``None`` when the mask is empty/too small to register on.
    """

    mask = load_mask(mask_path)
    if mask.shape != depth_shape:
        import cv2

        mask = (
            cv2.resize(
                mask.astype(np.uint8),
                (depth_shape[1], depth_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
    if int(mask.sum()) < MIN_MASK_AREA_PX:
        return None
    return mask


class _FrameProcessor:
    """Shared per-frame pose computation + artifact writing for one run."""

    def __init__(
        self,
        args: PoseEstimationVideoArgs,
        estimator: FoundationPoseEstimator,
        geometry_manifest: dict,
        masks_manifest: dict,
        poses_dir: Path,
        mesh=None,
        video_writer: PoseVideoWriter | None = None,
        shared_renderer: dict | None = None,
    ) -> None:
        self.args = args
        self.estimator = estimator
        self.geometry_manifest = geometry_manifest
        self.masks_manifest = masks_manifest
        self.poses_dir = poses_dir
        self.mesh = mesh
        self.video_writer = video_writer
        self._previous_object_center: np.ndarray | None = None
        # Mutable holder shared across processors so the GL offscreen renderer
        # is created at most once per run (it needs a live GL context).
        self._shared_renderer = shared_renderer if shared_renderer is not None else {}

    def _frame_inputs(self, frame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(rgb, depth, intrinsics) for a frame, or None when unusable."""

        geometry_dir = _geometry_frame_dir(self.geometry_manifest, frame.index)
        if geometry_dir is None:
            logger.info("[pose] frame {}: geometry missing, skip", frame.index)
            return None
        depth = _depth_from_geometry(geometry_dir)
        intrinsics = load_intrinsics(geometry_dir / INTRINSICS_FILENAME)
        rgb = load_rgb_image(frame.path)
        return rgb, depth, intrinsics

    def _object_center(self, frame, depth: np.ndarray) -> np.ndarray | None:
        mask_path = _mask_path(self.masks_manifest, frame.index, self.args.prompt_id)
        if mask_path is None:
            return None
        mask = _load_registration_mask(mask_path, depth.shape[:2])
        if mask is None:
            return None
        geometry_dir = _geometry_frame_dir(self.geometry_manifest, frame.index)
        if geometry_dir is None:
            return None
        points = np.load(geometry_dir / POINTS_FILENAME)
        if points.shape[:2] != mask.shape:
            return None
        observed = np.asarray(points[mask], dtype=np.float64)
        observed = observed[np.isfinite(observed).all(axis=1)]
        if len(observed) < MIN_MASK_AREA_PX:
            return None
        return np.median(observed, axis=0)

    def reset_translation_prior(self, frame) -> None:
        inputs = self._frame_inputs(frame)
        self._previous_object_center = (
            self._object_center(frame, inputs[1]) if inputs is not None else None
        )

    def _write_outputs(self, frame, pose: np.ndarray, rgb, intrinsics) -> str:
        pose_filename = POSE_FILENAME_PATTERN.format(frame.index)
        np.savetxt(self.poses_dir / pose_filename, np.asarray(pose, dtype=np.float64))
        if self.video_writer is not None and self.mesh is not None:
            if "renderer" not in self._shared_renderer:
                # Headless GL: pick EGL before PyOpenGL/pyrender is imported.
                os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
                from offscreen_renderer import ModelRendererOffscreen

                h, w = rgb.shape[:2]
                self._shared_renderer["renderer"] = ModelRendererOffscreen(
                    intrinsics, h, w
                )
            vis = render_mesh_overlay(
                frame_rgb=rgb,
                pose=np.asarray(pose, dtype=np.float64),
                mesh=self.mesh,
                intrinsics=intrinsics,
                renderer=self._shared_renderer["renderer"],
            )
            self.video_writer.add(frame.index, vis)
        return pose_filename

    def register_frame(self, frame, *, anchor_frame: int | None = None) -> PoseEntry:
        """Full registration; the entry carries the registered pose forward."""

        inputs = self._frame_inputs(frame)
        if inputs is None:
            return _skipped_entry(frame)
        rgb, depth, intrinsics = inputs

        mask_path = _mask_path(
            self.masks_manifest,
            frame.index,
            self.args.prompt_id,
        )
        mask = (
            _load_registration_mask(mask_path, depth.shape[:2])
            if mask_path is not None
            else None
        )
        if mask is None:
            logger.warning("[pose] frame {}: no usable mask, skipped", frame.index)
            return _skipped_entry(frame)

        pose = self.estimator.register(
            rgb=rgb, depth=depth, mask=mask, intrinsics=intrinsics
        )
        pose_filename = self._write_outputs(frame, pose, rgb, intrinsics)
        return PoseEntry(
            index=frame.index,
            frame_filename=frame.frame_filename,
            timestamp_sec=frame.timestamp_sec,
            pose_filename=pose_filename,
            tracked=True,
            method="register-anchor" if anchor_frame is not None else "register",
            direction="register",
            anchor_frame=anchor_frame,
        )

    def seed_frame(
        self,
        frame,
        pose: np.ndarray,
        *,
        anchor_frame: int,
    ) -> PoseEntry:
        """Seed exactly from an obj_recon pose without global registration."""

        inputs = self._frame_inputs(frame)
        if inputs is None:
            return _skipped_entry(frame)
        rgb, depth, intrinsics = inputs
        self.estimator.set_pose(pose)
        refined_pose = np.asarray(
            self.estimator.track(rgb=rgb, depth=depth, intrinsics=intrinsics)
        )
        pose_filename = self._write_outputs(frame, refined_pose, rgb, intrinsics)
        return PoseEntry(
            index=frame.index,
            frame_filename=frame.frame_filename,
            timestamp_sec=frame.timestamp_sec,
            pose_filename=pose_filename,
            tracked=True,
            method="obj-recon-seed-refine",
            direction="register",
            anchor_frame=anchor_frame,
        )

    def track_frame(
        self,
        frame,
        *,
        backwards: bool,
        anchor_frame: int | None = None,
    ) -> PoseEntry:
        """Track from the estimator's current pose into this frame."""

        inputs = self._frame_inputs(frame)
        if inputs is None:
            return _skipped_entry(frame)
        rgb, depth, intrinsics = inputs

        object_center = self._object_center(frame, depth)
        if (
            self.args.translation_prior
            and self._previous_object_center is not None
            and object_center is not None
        ):
            delta = object_center - self._previous_object_center
            if np.linalg.norm(delta) <= self.args.max_translation_step_m:
                self.estimator.shift_pose_translation(delta)
            else:
                logger.warning(
                    "[pose] frame {}: reject {:.3f} m translation prior step",
                    frame.index,
                    float(np.linalg.norm(delta)),
                )
        self._previous_object_center = object_center

        # FoundationPose remains responsible for the rigid refinement, including
        # rotation. The point-map prior only keeps rapid translation inside its crop.
        pose = np.asarray(
            self.estimator.track(rgb=rgb, depth=depth, intrinsics=intrinsics)
        )
        if (
            self.args.anchor_translation_to_geometry
            and object_center is not None
            and self.mesh is not None
        ):
            pose = pose.copy()
            pose[:3, 3] = object_center - pose[:3, :3] @ np.asarray(
                self.mesh.centroid, dtype=np.float64
            )
            self.estimator.set_pose(pose)

        # Corner case: overlapping symmetric silhouettes can make both branches
        # converge nominally; the manifest records direction so downstream can
        # sanity-check drift between the two passes at the boundary.

        pose_filename = self._write_outputs(frame, pose, rgb, intrinsics)
        return PoseEntry(
            index=frame.index,
            frame_filename=frame.frame_filename,
            timestamp_sec=frame.timestamp_sec,
            pose_filename=pose_filename,
            tracked=True,
            method="track-backward" if backwards else "track",
            direction="backward" if backwards else "forward",
            anchor_frame=anchor_frame,
        )


def run_video_pose_estimation(
    args: PoseEstimationVideoArgs,
    estimator: FoundationPoseEstimator | None = None,
) -> PoseEstimationOutputs:
    """Register on the init frame, then track the object through the video."""

    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")
    if args.reinit_every < 0:
        raise ValueError("--reinit-every must be >= 0.")

    frames_json = args.frames_json.expanduser().resolve()
    manifest = load_frame_manifest(frames_json)

    masks_manifest = load_masks_manifest(args.masks_json)
    geometry_json = args.geometry_json
    if geometry_json is None:
        clip_root_guess = frames_json.parent.parent
        geometry_json = clip_root_guess / "geometry" / "geometry.json"
    geometry_manifest = load_geometry_manifest(geometry_json)

    mesh_path = args.mesh_path.expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Object mesh not found: {mesh_path}")
    object_name = args.object_name or mesh_path.stem
    if args.prompt_id is None:
        args.prompt_id = object_name.replace("_", " ")

    clip_stem = frames_json.parent.parent.name
    if args.output_root is None:
        # Unified pipeline layout: derive the outputs root from frames_json so
        # this stage lands next to process/segment/geometry under
        # ``outputs/<clip>/`` regardless of the current working directory.
        output_root = frames_json.parent.parent.parent
    else:
        output_root = args.output_root.expanduser().resolve()
    clip_root = output_root / clip_stem
    stage_dir = clip_root / "pose_estimation"
    poses_dir = stage_dir / "poses"

    if args.foundationpose.overwrite and poses_dir.exists():
        shutil.rmtree(poses_dir)
    poses_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    selected = manifest.entries
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    selected_indices = {frame.index for frame in selected}
    mv_anchor_list = _load_mv_anchor_poses(mesh_path)
    mv_anchor_by_frame = {anchor.frame_index: anchor for anchor in mv_anchor_list}
    if mv_anchor_list and args.anchor_frames is not None:
        logger.warning("[pose] --anchor-frames is ignored for an MV mesh")
    if mv_anchor_list and args.reinit_every:
        logger.warning("[pose] --reinit-every is ignored for an MV mesh")
    resolved_anchors = _resolve_anchor_frames(mesh_path, args.anchor_frames)
    if not resolved_anchors:
        resolved_anchors = [args.init_frame]
    missing_anchors = [
        index for index in resolved_anchors if index not in selected_indices
    ]
    if missing_anchors:
        raise ValueError(
            f"Anchor frames not in the selected frame manifest: {missing_anchors}"
        )
    anchors = set(resolved_anchors)
    first_anchor_position = next(
        i for i, frame in enumerate(selected) if frame.index == resolved_anchors[0]
    )
    backward_frames = [frame for frame in reversed(selected[:first_anchor_position])]

    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")

    # SAM3D meshes are in normalized units; FoundationPose never fits cross-view
    # scale, so it must be handed a metric mesh (the refiner then optimises
    # rotation + depth where the depth comes right from the pointmap). The
    # scale comes from the obj_recon stage's own layout.json (SAM3D's metric
    # fit) unless overridden explicitly.
    mesh_scale = args.mesh_scale
    layout_path = _find_layout_path(mesh_path) if args.start_from_layout else None
    if mesh_scale is None and args.start_from_layout:
        if layout_path is None:
            raise FileNotFoundError(
                "No obj_recon layout.json found next to the mesh or its parent. "
                "Pass --mesh-scale explicitly or --no-start-from-layout."
            )
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        scale_xyz = layout["objects"][0]["local_to_scene"]["scale"]
        mesh_scale = float(np.mean(scale_xyz))
        logger.info("[pose] mesh scale from {}: {:.4f} m/unit", layout_path, mesh_scale)
    if mv_anchor_list and mesh_scale is not None:
        anchor_scales = np.asarray([anchor.scale for anchor in mv_anchor_list])
        if not np.allclose(anchor_scales, mesh_scale, rtol=0.05, atol=1e-6):
            raise ValueError(
                "MV per-view scales disagree with the mesh scale: "
                f"mesh={mesh_scale}, views={anchor_scales.tolist()}"
            )
    if mesh_scale is not None and mesh_scale != 1.0:
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices * mesh_scale
        logger.info(
            "[pose] tracking in metric units: mesh extent {:.3f} m",
            float(np.max(mesh.vertices.max(0) - mesh.vertices.min(0))),
        )

    active_estimator = estimator or FoundationPoseEstimator(mesh, args.foundationpose)

    video_writer = (
        PoseVideoWriter(stage_dir / "vis.mp4", manifest.fps) if args.vis else None
    )
    # Both processors share one offscreen renderer (single GL context) and one
    # video writer; frames are buffered and flushed in index order on close.
    shared_renderer: dict = {}

    processor = _FrameProcessor(
        args,
        active_estimator,
        geometry_manifest,
        masks_manifest,
        poses_dir,
        mesh=mesh,
        video_writer=video_writer,
        shared_renderer=shared_renderer,
    )

    entries: list[PoseEntry] = []
    first_anchor_pose: np.ndarray | None = None
    active_anchor: int | None = None
    forward_ok = False

    # An MV reconstruction provides one authoritative seed: the middle view and
    # its fitted metric pose. FoundationPose then follows the actual observations
    # frame by frame in both temporal directions; no pose interpolation is used.
    for frame in selected[first_anchor_position:]:
        is_anchor = frame.index in anchors
        is_periodic_reinit = (
            not mv_anchor_list
            and args.reinit_every > 0
            and (frame.index % args.reinit_every) == 0
        )
        if is_anchor and frame.index in mv_anchor_by_frame:
            target_pose = mv_anchor_by_frame[frame.index].object_to_camera
            entry = processor.seed_frame(
                frame,
                target_pose,
                anchor_frame=frame.index,
            )
            forward_ok = entry.tracked
            active_anchor = frame.index if entry.tracked else active_anchor
            if entry.tracked:
                first_anchor_pose = np.loadtxt(poses_dir / entry.pose_filename).reshape(
                    4, 4
                )
                processor.reset_translation_prior(frame)
        elif is_anchor or is_periodic_reinit:
            anchor = frame.index if is_anchor else active_anchor
            entry = processor.register_frame(frame, anchor_frame=anchor)
            if entry.tracked:
                forward_ok = True
                active_anchor = frame.index
                if frame.index == resolved_anchors[0]:
                    first_anchor_pose = np.loadtxt(
                        poses_dir / entry.pose_filename
                    ).reshape(4, 4)
            elif forward_ok:
                logger.warning(
                    "[pose] anchor {} registration failed; continuing previous chain",
                    frame.index,
                )
                entry = processor.track_frame(
                    frame,
                    backwards=False,
                    anchor_frame=active_anchor,
                )
        elif forward_ok:
            entry = processor.track_frame(
                frame,
                backwards=False,
                anchor_frame=active_anchor,
            )
        else:
            entry = _skipped_entry(frame)
        entries.append(entry)

    # Seed the same estimator with the first anchor pose and track toward frame 0.
    # Reusing it avoids loading another scorer/refiner pair into GPU memory.
    if not backward_frames:
        logger.info(
            "[pose] first anchor {} is the first selected frame: no backward tracking needed",
            resolved_anchors[0],
        )
    elif first_anchor_pose is not None:
        active_estimator.set_pose(first_anchor_pose)
        processor.reset_translation_prior(selected[first_anchor_position])
        for frame in backward_frames:
            entries.append(
                processor.track_frame(
                    frame,
                    backwards=True,
                    anchor_frame=resolved_anchors[0],
                )
            )
    else:
        for frame in backward_frames:
            entries.append(_skipped_entry(frame))

    entries.sort(key=lambda entry: entry.index)

    vis_video_path = video_writer.close() if video_writer is not None else None
    if vis_video_path is not None:
        logger.info("[pose] vis overlay video: {}", vis_video_path)

    _write_json(
        stage_dir / "poses.json",
        _manifest_dict(
            args,
            manifest,
            entries,
            anchors=resolved_anchors,
            layout_path=layout_path,
            mesh_scale=mesh_scale,
            poses_dir=poses_dir,
        ),
    )
    _write_json(stage_dir / "config.json", _config_dict(args, manifest))

    tracked = sum(1 for entry in entries if entry.tracked)
    logger.info(
        "[pose] Done: object={} frames={} tracked={} anchors={} out={}",
        object_name,
        len(entries),
        tracked,
        resolved_anchors,
        stage_dir,
    )

    filtered_poses_json_path: Path | None = None
    temporal_filter_args = args.temporal_filter
    if temporal_filter_args.enabled:
        mesh_extent = float(
            np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
        )
        filtered_poses_json_path = apply_temporal_filter_to_run(
            stage_dir,
            temporal_filter_args,
            mesh_extent_m=mesh_extent,
            track_refine_iter=args.foundationpose.track_refine_iter,
            masks_manifest=masks_manifest,
            geometry_manifest=geometry_manifest,
            prompt_id=args.prompt_id,
        )
        logger.info("[pose] temporal filter: {}", filtered_poses_json_path)

    return PoseEstimationOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        poses_dir=poses_dir,
        vis_video_path=vis_video_path,
        poses_json_path=stage_dir / "poses.json",
        config_json_path=stage_dir / "config.json",
        entries=entries,
        filtered_poses_json_path=filtered_poses_json_path,
    )


def _manifest_dict(
    args: PoseEstimationVideoArgs,
    manifest: FrameManifest,
    entries: list[PoseEntry],
    *,
    anchors: list[int],
    layout_path: Path | None,
    mesh_scale: float | None,
    poses_dir: Path,
) -> dict:
    return {
        "schema_version": "2.0",
        "stage": "pose_estimation",
        "source_frames_json": str(args.frames_json.expanduser().resolve()),
        "source_masks_json": str(args.masks_json.expanduser().resolve()),
        "source_video": manifest.source_video,
        "fps": manifest.fps,
        "width": manifest.width,
        "height": manifest.height,
        "frame_format": manifest.format,
        "frame_count": manifest.frame_count,
        "processed_count": len(entries),
        "mesh_path": str(args.mesh_path.expanduser().resolve()),
        "object_name": args.object_name or args.mesh_path.stem,
        "init_frame": args.init_frame,
        "anchor_frames": anchors,
        "seed_frame_index": anchors[0],
        "prompt_id": args.prompt_id,
        "reinit_every": args.reinit_every,
        "layout_json": str(layout_path) if layout_path is not None else None,
        "source_view_poses_json": (
            str(args.mesh_path.expanduser().resolve().parent / "view_poses.json")
            if _is_mv_mesh(args.mesh_path)
            else None
        ),
        "mesh_scale": mesh_scale,
        "poses_dir": str(poses_dir.resolve()),
        "entries": [entry.to_dict() for entry in entries],
    }


def _config_dict(args: PoseEstimationVideoArgs, manifest: FrameManifest) -> dict:
    return {
        "package": {"name": "pose_estimation", "version": __version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "frames_json": str(args.frames_json.expanduser().resolve()),
            "masks_json": str(args.masks_json.expanduser().resolve()),
            "mesh_path": str(args.mesh_path.expanduser().resolve()),
            "source_video": manifest.source_video,
            "fps": manifest.fps,
            "width": manifest.width,
            "height": manifest.height,
            "frame_format": manifest.format,
            "frame_count": manifest.frame_count,
        },
        "pose_estimation": {
            "weights_root": str(args.foundationpose.weights_root.expanduser()),
            "device": args.foundationpose.device,
            "est_refine_iter": args.foundationpose.est_refine_iter,
            "track_refine_iter": args.foundationpose.track_refine_iter,
            "track_crop_ratio": args.foundationpose.track_crop_ratio,
            "init_frame": args.init_frame,
            "anchor_frames": args.anchor_frames,
            "prompt_id": args.prompt_id,
            "reinit_every": args.reinit_every,
            "translation_prior": args.translation_prior,
            "max_translation_step_m": args.max_translation_step_m,
            "anchor_translation_to_geometry": args.anchor_translation_to_geometry,
            "mesh_scale": args.mesh_scale,
            "start_from_layout": args.start_from_layout,
            "max_frames": args.max_frames,
            "temporal_filter": vars(args.temporal_filter),
            "vis": args.vis,
            "debug": args.foundationpose.debug,
        },
        "software": {},
    }
