"""Mesh reconstruction from RGB frames and segment-stage masks using SAM 3D Objects.

This module joins the ``segment`` stage's ``masks.json`` with the ``geometry``
stage's ``geometry.json`` by frame index. Segment supplies one binary mask per
tracked prompt; geometry supplies the precomputed MoGe point map. The SAM 3D
Objects pipeline is loaded once, with its internal depth model disabled, and
reused across all frames.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import trimesh
from loguru import logger
from PIL import Image

from obj_recon import __version__
from obj_recon.device import resolve_torch_device, set_cuda_device_if_indexed
from obj_recon.media import load_mask, load_rgba_image, mask_is_empty, mask_to_alpha

# Coordinate conversion matrices used by generate_mesh_sam3d.py
P3D_TO_ISAAC = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32).T


@dataclass
class MeshReconArgs:
    """Arguments for SAM 3D Objects mesh reconstruction."""

    config: Path = Path(__file__).parents[2] / "weights/sam3d/pipeline.yaml"
    """Path to the SAM 3D Objects inference pipeline YAML config (weights live
    alongside it in ``weights/sam3d/``)."""

    device: str = "auto"
    """Torch device: ``"auto"``, ``"cuda"``, ``"cpu"``, or a concrete index like ``"cuda:0"``."""

    seed: int = 42
    """Base random seed for the diffusion pipeline."""

    with_mesh_postprocess: bool = True
    """Apply mesh postprocessing (smoothing, hole filling)."""

    with_texture_baking: bool = True
    """Bake vertex colors into a texture atlas."""

    use_vertex_color: bool = False
    """Use vertex colors instead of texture baking."""

    with_layout_postprocess: bool = True
    """Refine pose and scale against the mask and geometry point map."""


@dataclass(frozen=True)
class GeometryFrame:
    """Precomputed point-map geometry for one process frame."""

    index: int
    frame_filename: str
    points_path: Path
    intrinsics_path: Path


@dataclass(frozen=True)
class MeshOutput:
    """One reconstructed object's mesh and its pose in the scene frame."""

    object_name: str
    """Prompt id (e.g. ``"yellow spoon"``)."""

    mesh_path: Path
    """Path to the saved ``.obj`` file."""

    rotation: torch.Tensor | None
    """Quaternion (w, x, y, z) in local-to-camera frame."""

    translation: torch.Tensor | None
    """Translation vector in local-to-camera frame."""

    scale: torch.Tensor | None
    """Uniform scale factor."""

    new_quat: np.ndarray | None
    """Quaternion (scalar-first) after procrustes alignment to the ISAAC frame."""

    alignment_iou: float | None
    """Mask/geometry alignment IoU after layout post-optimization."""

    alignment_iou_before: float | None
    """Alignment IoU before layout post-optimization, when reported."""

    alignment_accepted: bool | None
    """Whether the GS layout optimizer accepted its refined pose."""


@dataclass(frozen=True)
class MeshReconEntry:
    """Per-frame record written into ``meshes.json``."""

    index: int
    """Frame index from the segment manifest entry."""

    frame_filename: str
    """Original frame filename (relative to the frames directory)."""

    frame_path: str
    """Resolved absolute path to the source RGB frame."""

    objects: tuple[MeshOutput, ...]
    """One entry per successfully reconstructed object in this frame."""

    def to_summary_dict(self) -> dict:
        return {
            "index": self.index,
            "frame_filename": self.frame_filename,
            "frame_path": self.frame_path,
            "objects": [
                {
                    "object_name": o.object_name,
                    "object_dir": o.mesh_path.parent.name,
                    "mesh_obj": o.mesh_path.name,
                    "alignment_iou": o.alignment_iou,
                    "alignment_iou_before": o.alignment_iou_before,
                    "alignment_accepted": o.alignment_accepted,
                }
                for o in self.objects
            ],
        }


@dataclass(frozen=True)
class MeshReconOutputs:
    """Everything produced by one mesh reconstruction run."""

    stage_dir: Path
    """Stage output directory (e.g. ``outputs/<clip>/obj_recon/``)."""

    meshes_dir: Path
    """Root directory holding one sub-folder per reconstructed frame."""

    meshes_json_path: Path
    """Path to ``meshes.json`` (the per-frame manifest)."""

    config_json_path: Path
    """Path to ``config.json`` (effective run configuration)."""

    entries: tuple[MeshReconEntry, ...]
    """One entry per processed frame."""


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sanitize_prompt_id(prompt_id: str) -> str:
    """Turn a prompt id like ``"yellow spoon"`` into a safe directory name."""
    safe = prompt_id.strip().lower().replace(" ", "_")
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in safe)


def _layout_has_objects(layout_path: Path) -> bool:
    """Return whether a prior frame output contains at least one object."""
    if not layout_path.exists():
        return False
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("objects"))


class _ExtendedInference:
    """Direct wrapper around ``InferencePipelinePointMap``.

    We deliberately bypass the upstream ``notebook/inference.py`` helper:
    that module is a Jupyter/Gradio demo whose top-level imports
    (``CONDA_PREFIX`` env juggling, ``seaborn``,
    ``sam3d_objects.utils.visualization`` → ``utils3d.numpy.depth_edge``)
    are either conda-only or belong to a newer ``utils3d`` than the
    pinned prebuilt-wheel set. None of it is needed for batch inference.
    Instantiating the pipeline class named by the checkpoint's
    ``pipeline.yaml`` (``_target_: InferencePipelinePointMap``) needs only
    packages we already ship, and no upstream file is modified.
    """

    def __init__(
        self,
        config_path: str,
        compile: bool = False,
        *,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_vertex_color: bool = False,
        with_layout_postprocess: bool = True,
    ):
        from pathlib import Path as _Path

        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # The vendored package's __init__ expects this flag even though its
        # optional init module is absent. Keep that upstream quirk local here.
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")

        # gsplat JIT-compiles its CUDA extension via torch's cpp_extension,
        # which resolves ``ninja`` through PATH (not the Python package).
        # A uv venv is usually not on PATH when invoked by absolute path, so
        # prepend the interpreter's own bin directory.
        import sys

        venv_bin = _Path(sys.executable).parent
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(venv_bin) not in path_entries:
            os.environ["PATH"] = os.pathsep.join([str(venv_bin), *path_entries])

        cfg = OmegaConf.load(str(config_path))
        cfg._target_ = "obj_recon.sam3d_pipeline.PrecomputedPointMapPipeline"
        cfg.workspace_dir = str(_Path(config_path).parent)
        cfg.compile_model = compile  # compile warmup requires an internal depth model
        cfg.depth_model = None

        self._pipeline = instantiate(cfg)
        self._with_mesh_postprocess = with_mesh_postprocess
        self._with_texture_baking = with_texture_baking
        self._use_vertex_color = use_vertex_color
        self._with_layout_postprocess = with_layout_postprocess

    def __call__(
        self,
        image: np.ndarray | Image.Image,
        pointmap: torch.Tensor,
        intrinsics: torch.Tensor,
        seed: int | None = None,
    ) -> dict:
        """Run SAM3D with geometry supplied by the external geometry stage."""
        if pointmap is None:
            raise ValueError(
                "A precomputed point map is required; run the geometry stage first."
            )
        result = self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=self._with_mesh_postprocess,
            with_texture_baking=self._with_texture_baking,
            with_layout_postprocess=self._with_layout_postprocess,
            use_vertex_color=self._use_vertex_color,
            stage1_inference_steps=None,
            pointmap={"points": pointmap, "intrinsics": intrinsics},
        )
        if self._with_layout_postprocess and "iou" not in result:
            raise RuntimeError(
                "SAM3D layout post-optimization did not return an aligned pose/scale."
            )
        return result


def _load_inference(
    config_path: str | Path,
    *,
    with_mesh_postprocess: bool = True,
    with_texture_baking: bool = True,
    use_vertex_color: bool = False,
    with_layout_postprocess: bool = True,
) -> _ExtendedInference:
    """Load the SAM 3D Objects inference pipeline with the given flags.

    Import is deferred so the package can be imported without CUDA available
    (the heavy import only happens when ``reconstruct_video`` is called).
    """
    logger.info("[SAM3D] Loading inference pipeline from: {}", config_path)
    return _ExtendedInference(
        str(config_path),
        compile=False,
        with_mesh_postprocess=with_mesh_postprocess,
        with_texture_baking=with_texture_baking,
        use_vertex_color=use_vertex_color,
        with_layout_postprocess=with_layout_postprocess,
    )


def _as_list(x) -> list[float]:
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return [float(v) for v in np.asarray(x).flatten().tolist()]


def _as_scalar(x, cast):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "item"):
        x = x.item()
    return cast(x)


def _save_layout_json(
    outputs: Sequence[MeshOutput],
    layout_path: Path,
) -> None:
    """Save per-object local-to-scene transforms for one frame to ``layout.json``.

    Mirrors the ``_save_layout_json`` helper from ``generate_mesh_sam3d.py``.
    """
    objects = []
    for i, out in enumerate(outputs):
        if out.rotation is None or out.translation is None or out.scale is None:
            continue

        q_wxyz = torch.as_tensor(_as_list(out.rotation))
        q_xyzw = torch.stack([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

        objects.append(
            {
                "index": i,
                "mesh_obj": out.mesh_path.name,
                "local_to_scene": {
                    "translation": _as_list(out.translation),
                    "scale": _as_list(out.scale),
                    "quat_wxyz": _as_list(out.rotation),
                    "new_quat": _as_list(out.new_quat)
                    if out.new_quat is not None
                    else None,
                    "quat_xyzw": _as_list(q_xyzw),
                },
            }
        )

    payload = {
        "frame": "sam3d_scene",
        "note": (
            "Transforms are local->scene as used by SAM3D make_scene/_export_scene_glb. "
            "Scale is typically uniform (same xyz). Absolute metric scale may be arbitrary."
        ),
        "objects": objects,
    }
    _write_json(layout_path, payload)


def _load_pointmap(points_path: Path, image_shape: tuple[int, int]) -> torch.Tensor:
    """Load a MoGe point map and convert it to PyTorch3D camera coordinates."""
    if not points_path.exists():
        raise FileNotFoundError(f"Geometry point map not found: {points_path}")

    points = np.load(points_path)
    expected_shape = (*image_shape, 3)
    if points.shape != expected_shape:
        raise ValueError(
            f"Point map shape mismatch: expected {expected_shape}, got {points.shape} "
            f"from {points_path}"
        )
    if not np.issubdtype(points.dtype, np.floating):
        raise ValueError(
            f"Point map must be floating point, got {points.dtype}: {points_path}"
        )

    pointmap = torch.from_numpy(points.astype(np.float32, copy=False))
    # MoGe uses OpenCV/R3 camera coordinates (x right, y down, z forward).
    # SAM3D's external pointmap branch expects PyTorch3D camera coordinates.
    pointmap = pointmap * pointmap.new_tensor([-1.0, -1.0, 1.0])
    return pointmap


def _load_intrinsics(
    intrinsics_path: Path,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Load pixel-space intrinsics and normalize them for SAM3D."""
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Geometry intrinsics not found: {intrinsics_path}")

    intrinsics = np.load(intrinsics_path)
    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"Expected 3x3 intrinsics, got {intrinsics.shape}: {intrinsics_path}"
        )
    if not np.issubdtype(intrinsics.dtype, np.floating):
        raise ValueError(
            f"Intrinsics must be floating point, got {intrinsics.dtype}: "
            f"{intrinsics_path}"
        )

    height, width = image_shape
    normalized = intrinsics.astype(np.float32, copy=True)
    normalized[0, :] /= width
    normalized[1, :] /= height
    return torch.from_numpy(normalized)


def _process_frame(
    frame_path: Path,
    mask_items: Sequence[tuple[str, Path]],  # [(prompt_id, mask_path)]
    geometry: GeometryFrame,
    frame_out_dir: Path,
    *,
    inference,
    seed: int,
) -> tuple[MeshOutput, ...]:
    """Reconstruct every (prompt, mask) pair for a single frame."""
    frame_out_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgba_image(frame_path)
    pointmap = _load_pointmap(geometry.points_path, image.shape[:2])
    intrinsics = _load_intrinsics(geometry.intrinsics_path, image.shape[:2])
    outputs: list[MeshOutput] = []

    for i, (prompt_id, mask_path) in enumerate(mask_items):
        if not mask_path.exists():
            logger.warning("  - Missing mask file, skip: {}", mask_path)
            continue

        mask = load_mask(mask_path)
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match frame shape "
                f"{image.shape[:2]}: {mask_path}"
            )
        if mask_is_empty(mask):
            logger.warning("  - Skipping empty mask: {} ({})", mask_path, prompt_id)
            continue

        obj_name = _sanitize_prompt_id(prompt_id)
        logger.info(
            "  - Reconstructing [{}/{}] {} <- {}",
            i + 1,
            len(mask_items),
            prompt_id,
            mask_path,
        )

        rgba = np.dstack([image[..., :3], mask_to_alpha(mask)])
        result = inference(
            Image.fromarray(rgba),
            pointmap,
            intrinsics,
            seed=seed + i,
        )

        mesh = result.get("glb")
        if mesh is None:
            logger.warning("  - No mesh produced for: {}", prompt_id)
            continue

        object_folder = frame_out_dir / obj_name
        object_folder.mkdir(parents=True, exist_ok=True)
        obj_path = object_folder / f"{obj_name}.obj"
        # Match the reference contract: export the canonical textured mesh.
        # Metric pose/scale stay in layout.json and must not also be baked into vertices.
        mesh.export(str(obj_path))
        logger.info("  Saved canonical mesh to: {}", obj_path)

        rotation = result.get("rotation")
        translation = result.get("translation")
        scale = result.get("scale")

        new_quat = None
        if rotation is not None and translation is not None and scale is not None:
            from pytorch3d.transforms import quaternion_to_matrix
            from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
            from scipy.spatial.transform import Rotation as R

            mesh_old_vertices = np.array(mesh.vertices, dtype=np.float32)
            vertices = mesh_old_vertices @ _R_YUP_TO_ZUP
            pose_device = rotation.device
            vertices_tensor = torch.from_numpy(vertices).float().to(pose_device)

            R_l2c = quaternion_to_matrix(rotation)
            l2c_transform = compose_transform(
                scale=scale, rotation=R_l2c, translation=translation
            )
            transformed = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
            new_mesh_vertices = transformed.squeeze(0).cpu().numpy() @ P3D_TO_ISAAC
            mesh.vertices = new_mesh_vertices

            matrix_proc, _, _ = trimesh.registration.procrustes(
                mesh_old_vertices,
                new_mesh_vertices,
                reflection=False,
                return_cost=True,
            )
            new_quat = R.from_matrix(matrix_proc[:3, :3]).as_quat(scalar_first=True)

        outputs.append(
            MeshOutput(
                object_name=prompt_id,
                mesh_path=obj_path,
                rotation=rotation,
                translation=translation,
                scale=scale,
                new_quat=new_quat,
                alignment_iou=_as_scalar(result.get("iou"), float),
                alignment_iou_before=_as_scalar(result.get("iou_before_optim"), float),
                alignment_accepted=_as_scalar(result.get("optim_accepted"), bool),
            )
        )

    _save_layout_json(outputs, frame_out_dir / "layout.json")
    return tuple(outputs)


def _select_prompt_masks(
    entry: dict,
    prompt_ids: Sequence[str] | None,
    masks_root: Path,
) -> list[tuple[str, Path]]:
    """Extract (prompt_id, absolute mask path) pairs from one manifest entry.

    Only prompt masks with ``has_mask`` are kept. If ``prompt_ids`` is given,
    only those prompts are kept (matched on ``prompt_id``).
    """
    wanted = set(prompt_ids) if prompt_ids else None
    items: list[tuple[str, Path]] = []
    for pm in entry.get("prompt_masks", []) or []:
        if not pm.get("has_mask"):
            continue
        pid = pm.get("prompt_id")
        if wanted is not None and pid not in wanted:
            continue
        mask_fn = pm.get("mask_filename")
        if not mask_fn:
            continue
        items.append((pid, masks_root / mask_fn))
    return items


def _resolve_prompt_ids(manifest: dict, requested: Sequence[str] | None) -> list[str]:
    """Decide which prompt ids to reconstruct.

    If ``requested`` is provided, use it verbatim (validated against the
    manifest). Otherwise use every prompt in the manifest whose
    ``input_type`` is ``"text"`` — i.e. reconstruct the *objects*, not the
    hands (hands already have meshes from the ``hand_recon`` stage).
    """
    available = [p["prompt_id"] for p in manifest.get("prompts", [])]
    if requested:
        unknown = [pid for pid in requested if pid not in available]
        if unknown:
            logger.warning(
                "[obj_recon] requested prompt_id(s) not in manifest: {} (available: {})",
                unknown,
                available,
            )
        return [pid for pid in requested if pid in available]

    text_prompts = [
        p["prompt_id"]
        for p in manifest.get("prompts", [])
        if p.get("input_type") == "text"
    ]
    if text_prompts:
        logger.info(
            "[obj_recon] no --prompt-id given; reconstructing text-prompt objects: {}",
            text_prompts,
        )
        return text_prompts
    return list(available)


def resolve_frame_path(
    entry: dict,
    *,
    frames_json: Path | None,
    masks_json: Path,
) -> Path:
    """Resolve the source RGB frame for one manifest entry.

    Resolution order:

    1. ``frames_json`` (process stage) -> ``frames_dir`` + same index;
    2. ``<masks_json.parent>/<source_frames_json>`` -> its ``frames_dir``;
    3. ``<masks_json.parent>/../process/frames`` (conventional layout).
    """
    idx = int(entry["index"])

    def _frame_from_frames_manifest(fj: Path, index: int) -> Path | None:
        data = json.loads(fj.read_text(encoding="utf-8"))
        frames_dir = Path(data.get("frames_dir", fj.parent / "frames"))
        if not frames_dir.is_absolute():
            frames_dir = fj.parent / frames_dir
        entries = data.get("entries", [])
        for e in entries:
            if int(e["index"]) == index:
                return (frames_dir / e["frame_filename"]).resolve()
        return None

    if frames_json is not None:
        frames_json = Path(frames_json).expanduser().resolve()
        if not frames_json.exists():
            raise FileNotFoundError(f"frames.json not found: {frames_json}")
        found = _frame_from_frames_manifest(frames_json, idx)
        if found is None:
            raise KeyError(
                f"frames.json has no entry for frame index {idx}: {frames_json}"
            )
        return found

    sibling = masks_json.parent / str(manifest_source_frames_json(masks_json))
    if frames_json is None and sibling.exists():
        found = _frame_from_frames_manifest(sibling, idx)
        if found is not None:
            return found

    conventional = (
        masks_json.parent.parent / "process" / "frames" / entry["frame_filename"]
    )
    if conventional.exists():
        return conventional

    raise FileNotFoundError(
        f"Could not resolve frame for entry index={idx} "
        f"(tried frames_json={frames_json}, sibling={sibling}, conventional={conventional})"
    )


def manifest_source_frames_json(masks_json: Path) -> str:
    """Return the ``source_frames_json`` field of a masks.json manifest."""
    try:
        data = json.loads(masks_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(data.get("source_frames_json", ""))


def load_geometry_manifest(geometry_json: Path) -> dict[int, GeometryFrame]:
    """Load and validate the geometry-stage manifest, indexed by frame index."""
    geometry_json = geometry_json.expanduser().resolve()
    if not geometry_json.exists():
        raise FileNotFoundError(f"geometry.json not found: {geometry_json}")

    payload = json.loads(geometry_json.read_text(encoding="utf-8"))
    if payload.get("stage") != "geometry":
        raise ValueError(f"Not a geometry manifest: {geometry_json}")

    frames_dir = Path(payload.get("frames_dir", geometry_json.parent / "frames"))
    if not frames_dir.is_absolute():
        frames_dir = geometry_json.parent / frames_dir

    frames: dict[int, GeometryFrame] = {}
    for entry in payload.get("entries", []):
        index = int(entry["index"])
        if index in frames:
            raise ValueError(f"Duplicate geometry frame index {index}: {geometry_json}")

        frame_dir = Path(entry.get("frame_dir", f"{index:06d}"))
        if not frame_dir.is_absolute():
            if frame_dir.parts and frame_dir.parts[0] == frames_dir.name:
                frame_dir = geometry_json.parent / frame_dir
            else:
                frame_dir = frames_dir / frame_dir
        points_path = frame_dir / entry["points"]
        intrinsics_path = frame_dir / entry["intrinsics"]
        frames[index] = GeometryFrame(
            index=index,
            frame_filename=entry["frame_filename"],
            points_path=points_path.resolve(),
            intrinsics_path=intrinsics_path.resolve(),
        )
    return frames


def reconstruct_video(
    masks_json: Path,
    *,
    geometry_json: Path,
    frames_json: Path | None = None,
    output_root: Path | None = None,
    prompt_ids: Sequence[str] | None = None,
    max_frames: int | None = None,
    frame_indices: Sequence[int] | None = None,
    skip_existing: bool = False,
    args: MeshReconArgs | None = None,
) -> MeshReconOutputs:
    """Run SAM 3D Objects over every frame of a ``segment`` manifest.

    This is the main entry point for the package. It joins segment masks with
    precomputed ``geometry`` point maps by frame index, loads the SAM3D
    pipeline once, then writes one ``meshes/<index>/`` folder per processed
    frame. SAM3D's internal MoGe model is disabled and never instantiated.

    The stage directory mirrors the other pipeline stages: given
    ``outputs/<clip>/segment/masks.json``, outputs land in
    ``outputs/<clip>/obj_recon/`` by default (or ``output_root/<clip>/obj_recon/``
    when ``output_root`` is provided).
    """
    args = args or MeshReconArgs()
    masks_json = Path(masks_json).expanduser().resolve()
    if not masks_json.exists():
        raise FileNotFoundError(f"masks.json not found: {masks_json}")

    manifest = json.loads(masks_json.read_text(encoding="utf-8"))
    if manifest.get("stage") != "segment":
        raise ValueError(f"Not a segment manifest: {masks_json}")
    if max_frames is not None and max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")

    geometry_json = Path(geometry_json).expanduser().resolve()
    geometry_frames = load_geometry_manifest(geometry_json)
    all_entries = manifest.get("entries", [])
    if frame_indices:
        requested_indices = set(frame_indices)
        available_indices = {int(entry["index"]) for entry in all_entries}
        unknown_indices = sorted(requested_indices - available_indices)
        if unknown_indices:
            raise ValueError(
                f"Requested frame indices are not in masks.json: {unknown_indices}"
            )
        all_entries = [
            entry for entry in all_entries if int(entry["index"]) in requested_indices
        ]
    if max_frames is not None:
        all_entries = all_entries[:max_frames]

    missing_geometry = [
        int(entry["index"])
        for entry in all_entries
        if int(entry["index"]) not in geometry_frames
    ]
    if missing_geometry:
        raise ValueError(
            "geometry.json does not cover selected segment frame indices: "
            f"{missing_geometry}. Run geometry with at least the same frame range."
        )
    for entry in all_entries:
        index = int(entry["index"])
        geometry_name = geometry_frames[index].frame_filename
        if geometry_name != entry["frame_filename"]:
            raise ValueError(
                f"Frame mismatch at index {index}: segment has "
                f"{entry['frame_filename']!r}, geometry has {geometry_name!r}."
            )

    # Determine output stage directory
    segment_dir = masks_json.parent
    clip_root = segment_dir.parent
    if output_root is not None:
        output_root = Path(output_root).expanduser().resolve()
        stage_dir = output_root / clip_root.name / "obj_recon"
    else:
        stage_dir = clip_root / "obj_recon"
    meshes_dir = stage_dir / "meshes"
    stage_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Which prompts to reconstruct
    selected = _resolve_prompt_ids(manifest, prompt_ids)
    if not selected:
        raise ValueError("No prompt ids selected for reconstruction.")

    # Resolve masks root (absolute in manifest, else relative to masks.json)
    masks_root = Path(manifest.get("masks_dir") or segment_dir / "masks")
    masks_root = (
        masks_root if masks_root.is_absolute() else (masks_json.parent / masks_root)
    )
    masks_root = masks_root.resolve()

    # Resolve and validate every lightweight input before loading the large model.
    plans = []
    for n, entry in enumerate(all_entries, start=1):
        idx = int(entry["index"])
        frame_fn = entry["frame_filename"]
        frame_out_dir = meshes_dir / f"{idx:06d}"

        if skip_existing and _layout_has_objects(frame_out_dir / "layout.json"):
            logger.info(
                "({}/{}) completed layout exists for frame {} -> SKIP",
                n,
                len(all_entries),
                frame_fn,
            )
            continue

        mask_items = _select_prompt_masks(
            entry, prompt_ids=selected, masks_root=masks_root
        )
        if not mask_items:
            logger.info(
                "({}/{}) no masks for selected prompts in frame {} -> SKIP",
                n,
                len(all_entries),
                frame_fn,
            )
            continue

        frame_path = resolve_frame_path(
            entry, frames_json=frames_json, masks_json=masks_json
        )
        geometry = geometry_frames[idx]
        for geometry_path in (geometry.points_path, geometry.intrinsics_path):
            if not geometry_path.exists():
                raise FileNotFoundError(f"Geometry artifact not found: {geometry_path}")
        plans.append(
            (n, idx, frame_fn, frame_path, mask_items, geometry, frame_out_dir)
        )

    inference = None
    if plans:
        device = resolve_torch_device(args.device)
        set_cuda_device_if_indexed(device)
        inference = _load_inference(
            args.config,
            with_mesh_postprocess=args.with_mesh_postprocess,
            with_texture_baking=args.with_texture_baking,
            use_vertex_color=args.use_vertex_color,
            with_layout_postprocess=args.with_layout_postprocess,
        )

    out_entries: list[MeshReconEntry] = []
    for n, idx, frame_fn, frame_path, mask_items, geometry, frame_out_dir in plans:
        logger.info(
            "({}/{}) frame={} pointmap={} objects={} out={}",
            n,
            len(all_entries),
            frame_path,
            geometry.points_path,
            [pid for pid, _ in mask_items],
            frame_out_dir,
        )

        objects = _process_frame(
            frame_path,
            mask_items,
            geometry,
            frame_out_dir,
            inference=inference,
            seed=args.seed + idx * 100,
        )

        out_entries.append(
            MeshReconEntry(
                index=idx,
                frame_filename=frame_fn,
                frame_path=str(frame_path),
                objects=objects,
            )
        )

    # Manifests
    _write_json(
        stage_dir / "meshes.json",
        {
            "schema_version": "1.0",
            "stage": "obj_recon",
            "source_masks_json": str(masks_json),
            "source_geometry_json": str(geometry_json),
            "source_frames_json": (
                str(frames_json.expanduser().resolve())
                if frames_json is not None
                else manifest_source_frames_json(masks_json)
            ),
            "prompt_ids": list(selected),
            "requested_frame_indices": list(frame_indices) if frame_indices else None,
            "meshes_dir": str(meshes_dir),
            "frame_count": len(all_entries),
            "processed_count": len(out_entries),
            "entries": [e.to_summary_dict() for e in out_entries],
        },
    )
    _write_json(
        stage_dir / "config.json",
        {
            "package": {"name": "obj_recon", "version": __version__},
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source": {
                "masks_json": str(masks_json),
                "geometry_json": str(geometry_json),
                "frames_json": str(frames_json) if frames_json else None,
                "prompt_ids": list(selected),
                "max_frames": max_frames,
                "frame_indices": list(frame_indices) if frame_indices else None,
            },
            "reconstruction": {
                "config": str(args.config.expanduser().resolve()),
                "device": args.device,
                "seed": args.seed,
                "with_mesh_postprocess": args.with_mesh_postprocess,
                "with_texture_baking": args.with_texture_baking,
                "use_vertex_color": args.use_vertex_color,
                "with_layout_postprocess": args.with_layout_postprocess,
                "processed_count": len(out_entries),
            },
        },
    )

    logger.info(
        "[obj_recon] Done: processed={}/{} frames out={}",
        len(out_entries),
        len(all_entries),
        stage_dir,
    )

    return MeshReconOutputs(
        stage_dir=stage_dir,
        meshes_dir=meshes_dir,
        meshes_json_path=stage_dir / "meshes.json",
        config_json_path=stage_dir / "config.json",
        entries=tuple(out_entries),
    )
