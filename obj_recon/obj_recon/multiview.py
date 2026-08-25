"""Multi-view object reconstruction via MV-SAM3D (vendored under ``pkgs/MV-SAM3D``).

Flows a small set of keyframes through ``InferencePipelinePointMap.run_multi_view``
with the default entropy-weighted fusion, reusing the SAME ``geometry``-stage
MoGe point maps the single-frame path consumes (no DA3 dependency). Emits

* one fused textured canonical mesh per object (``mesh.glb`` + ``mesh.obj``), and
* a reference keyframe metric pose (view-0 ``layout.json``) consumed by
  ``pose_estimation`` the same way the single-frame output is.

Installation note
-----------------
``obj_recon`` points its single ``sam3d_objects`` dependency at
``pkgs/MV-SAM3D``: it declares the same project / top-level package as upstream,
so both checkouts cannot coexist as two pip deps, and MV-SAM3D is a superset
(adds `pipeline.multi_view_*`, `pose_align`) with unchanged single-frame
behavior.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from obj_recon import __version__
from obj_recon.media import load_mask, load_rgba_image, mask_is_empty, mask_to_alpha
from obj_recon.reconstruct import (
    MeshReconArgs,
    _as_list,
    _sanitize_prompt_id,
    load_geometry_manifest,
    resolve_frame_path,
)

DEFAULT_MAX_VIEWS = 8


@dataclass
class MultiViewArgs:
    """MV-SAM3D multi-view reconstruction settings."""

    enabled: bool = False
    """Run multi-view (MV-SAM3D) reconstruction instead of per-frame single-view.
    Fuses several keyframes into one mesh per object; ``frame_index`` /
    ``keyframe_strategy`` / ``num_views`` select the views."""

    keyframe_strategy: str = "even"
    """How to pick keyframes: ``manual`` (use ``frame_index`` verbatim, in given
    order), ``even`` (evenly spaced over the segment range), or ``ffprobe``
    (scene/I-frames from the source video via ffprobe, falling back to even)."""

    num_views: int = 7
    """Number of keyframes for ``even`` / ``ffprobe`` strategies (>=2 for a
    multi-view run). The odd default gives pose tracking one unambiguous temporal
    middle view while retaining dense coverage of the clip."""

    max_views_cap: int = DEFAULT_MAX_VIEWS
    """Hard cap on the number of views fed to the diffusion model."""

    keyframe_video: Path | None = None
    """Source video for the ``ffprobe`` strategy; defaults to
    ``source_video`` / ``video`` fields in the frames.json manifest."""

    seed: int = 42
    stage1_inference_steps: int | None = None
    stage2_inference_steps: int | None = None
    mode: str = "multidiffusion"
    ss_weighting: bool = True
    ss_entropy_layer: int = 9
    ss_entropy_alpha: float = 60.0
    stage2_weighting: bool = True
    stage2_entropy_alpha: float = 30.0
    stage2_attention_layer: int = 6
    stage2_attention_step: int = 1
    stage2_min_weight: float = 0.001

    min_fit_iou: float = 0.2
    """Minimum silhouette IoU for accepting per-view metric post-optimization.
    Lower-scoring fits fall back to the decoded MV pose rather than poisoning
    the FoundationPose seed."""


def _evenly_spaced(candidates: Sequence[int], n: int) -> list[int]:
    """Pick ``n`` evenly spaced indices from a sorted candidate list."""
    cands = list(candidates)
    if len(cands) <= n:
        return cands
    pos = np.linspace(0, len(cands) - 1, num=n).round().astype(int)
    out = sorted({cands[p] for p in pos})
    return out


def _ffprobe_keyframes(video: Path, candidates: Sequence[int]) -> list[int] | None:
    """Return the sorted candidate indices closest to the video's keyframes.

    Uses ffprobe scene/score//I-frame detection; returns ``None`` when the
    video or ffprobe is unavailable or no keyframes are found (caller falls
    back to even spacing).
    """
    video = Path(video).expanduser()
    if not video.exists():
        logger.warning("[mv] ffprobe video not found: {}", video)
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pict_type,best_effort_timestamp_time",
                "-of",
                "csv=p=0",
                str(video),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("[mv] ffprobe failed ({exc}); falling back to even", exc=exc)
        return None
    if proc.returncode != 0:
        logger.warning("[mv] ffprobe exited {}: {}", proc.returncode, proc.stderr[:200])
        return None
    iframe_idx: list[int] = []
    for i, line in enumerate(proc.stdout.splitlines()):
        # csv lines are "<time>,<pict_type>"; I-frames carry pict_type=I
        if line.rstrip().endswith(",I"):
            iframe_idx.append(i)
    if not iframe_idx:
        return None
    cands = sorted(candidates)
    # Snap each I-frame position to the nearest candidate index (deduped).
    snapped = sorted({min(cands, key=lambda c, t=i: abs(c - t)) for i in iframe_idx})
    return snapped


def select_keyframes(
    all_indices: Sequence[int],
    *,
    strategy: str,
    num_views: int,
    max_views_cap: int,
    manual: Sequence[int] | None = None,
    video: Path | None = None,
) -> list[int]:
    """Resolve the ordered keyframe list for a multi-view run.

    ``all_indices`` is the sorted set of frame indices that have BOTH a segment
    mask and a geometry point map. The first returned index doubles as the
    reference (view-0) frame whose pose is exported.
    """
    n = max(2, int(num_views))
    cap = max(2, int(max_views_cap))
    strategy = (strategy or "even").lower()

    if strategy == "manual":
        if not manual:
            raise ValueError("--keyframe-strategy manual requires --frame-index list.")
        order = {idx: k for k, idx in enumerate(manual)}
        sel = [i for i in manual if i in set(all_indices)]
        missing = [i for i in manual if i not in set(all_indices)]
        if missing:
            logger.warning(
                "[mv] manual keyframes without mask/geometry ignored: {}", missing
            )
        if len(sel) < 2:
            raise ValueError(
                f"manual keyframes must contain >=2 usable frames, got {sel} "
                f"(n_usable={len(all_indices)})"
            )
        sel = sorted(sel, key=lambda i: order[i])  # preserve user order (view 0 first)
    elif strategy == "ffprobe":
        snapped = _ffprobe_keyframes(video, all_indices) if video else None
        if snapped:
            sel = snapped
            logger.info("[mv] ffprobe keyframes (snapped to candidates): {}", sel)
        else:
            logger.warning(
                "[mv] ffprobe produced no keyframes; falling back to even spacing"
            )
            sel = _evenly_spaced(all_indices, n)
    else:  # even
        sel = _evenly_spaced(all_indices, n)

    if len(sel) > cap:
        sel = _evenly_spaced(sel, cap)
    if len(sel) < 2:
        raise ValueError(f"Multi-view needs >=2 keyframes, resolved {sel}")
    return sel


def _patch_mv_compute_pointmap_intrinsics(pipeline) -> None:
    """Inject geometry-stage intrinsics into MV-SAM3D's external-pointmap branch.

    MV's ``compute_pointmap`` sets ``intrinsics = None`` for external point maps
    and re-infers them via ``infer_intrinsics_from_pointmap`` (MoGe's
    ``recover_focal_shift`` least-squares). On MoGe's own point maps that solver
    hits z+shift=0 → NaN → "Residuals are not finite". We wrap the method so
    the queued per-view intrinsics are consumed in order, skipping the solver.
    """
    import types

    pending: list = []

    # Capture the unbound original once (avoid a leaky recursion reference).
    orig = type(pipeline).compute_pointmap
    _marker = "_mv_intrinsics_patched"
    if getattr(pipeline, _marker, False):
        return

    def compute_pointmap(self, image, pointmap=None):
        self._mv_skip_infer = bool(pending)
        try:
            return orig(self, image, pointmap=pointmap)
        finally:
            self._mv_skip_infer = False

    # Bypass the solver by monkeypatching the pipeline's module-level helper:
    # while a geometry intrinsics is queued (``_mv_skip_infer`` is True inside
    # the call above), return it instead of running MoGe's recover_focal_shift
    # (which NaNs on z+shift=0 for MoGe point maps). pop() consumes in order.
    # compute_pointmap does a bare call ``infer_intrinsics_from_pointmap(...)``
    # inside pkgs/MV-SAM3D's inference_pipeline_pointmap module. Bare calls bind
    # via the module's globals ONLY (class attrs are not consulted), so we patch
    # the module global — the only place that actually intercepts it.
    import sam3d_objects.pipeline.inference_pipeline_pointmap as ipm

    orig_infer = ipm.infer_intrinsics_from_pointmap

    def _stub(points, device=None):
        active = getattr(pipeline, "_mv_skip_infer", False)
        if active and pending:
            intr = pending.pop(0).to(device or "cpu")
            logger.info(
                "[mv]   injected geometry intrinsics (fx_norm={:.3f})",
                float(intr[0, 0]),
            )
            return {"intrinsics": intr}
        return orig_infer(points, device=device)

    ipm.infer_intrinsics_from_pointmap = _stub

    pipeline.compute_pointmap = types.MethodType(compute_pointmap, pipeline)
    pipeline._mv_pending_intrinsics = pending
    setattr(pipeline, _marker, True)


def _load_mv_inference(args: MeshReconArgs):
    """Instantiate the MV-SAM3D point-map pipeline (depth model disabled).

    Mirrors ``reconstruct._load_inference`` but targets the MV checkout's
    default ``compute_pointmap``, which accepts external OpenCV-convention
    point maps (geometry MoGe points feed through unchanged).
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")

    # gsplat JIT-compiles its CUDA extension via torch's cpp_extension, which
    # resolves ``ninja`` through PATH; prepend the interpreter's bin dir.
    import sys

    venv_bin = Path(sys.executable).parent
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(venv_bin) not in path_entries:
        os.environ["PATH"] = os.pathsep.join([str(venv_bin), *path_entries])

    cfg = OmegaConf.load(str(args.config))
    cfg.workspace_dir = str(Path(args.config).parent)
    cfg.compile_model = False
    cfg.depth_model = None  # external geometry point maps only

    logger.info("[mv] Loading MV-SAM3D inference pipeline from: {}", args.config)
    pipeline = instantiate(cfg)
    if pipeline.layout_post_optimization_method is None:
        from sam3d_objects.pipeline.inference_utils import layout_post_optimization

        pipeline.layout_post_optimization_method = layout_post_optimization
    return pipeline


def _mv_pointmap(
    points: np.ndarray, mask: np.ndarray, downsample: int = 2
) -> np.ndarray:
    """Pack a geometry point map (H,W,3, OpenCV) as (3,H',W') for run_multi_view."""
    pts = points.astype(np.float32, copy=True)
    if downsample > 1:
        h, w = pts.shape[:2]
        hs, ws = h // downsample, w // downsample
        mask_t = torch.from_numpy(mask[None, None].astype(np.float32))
        mask_s = torch.nn.functional.interpolate(mask_t, size=(hs, ws), mode="nearest")[
            0, 0
        ].numpy()
        pts = pts[::downsample, ::downsample]
        mask = mask_s > 0
    # Zero out background so invalid points don't pollute the conditioning.
    pts[~mask] = 0.0
    return pts.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)


def _resize_rgba(rgba: np.ndarray, downsample: int) -> np.ndarray:
    if downsample <= 1:
        return rgba
    h, w = rgba.shape[:2]
    return np.asarray(
        Image.fromarray(rgba).resize(
            (w // downsample, h // downsample), Image.Resampling.BILINEAR
        )
    )


def _normalize_intrinsics(
    intrinsics: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Convert pixel intrinsics to resolution-independent normalized values."""

    normalized = intrinsics.astype(np.float32, copy=True)
    normalized[0, :] /= width
    normalized[1, :] /= height
    return normalized


def _save_layout(
    out_dir: Path,
    name: str,
    rotation,
    translation,
    scale,
    *,
    reference_frame_index: int,
) -> None:
    """Save the fused mesh's reference layout (same schema as single-frame)."""
    q_wxyz = torch.as_tensor(_as_list(rotation))
    q_xyzw = torch.stack([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).tolist()
    payload = {
        "schema_version": "2.0",
        "frame": "sam3d_pytorch3d_camera",
        "reference_frame_index": reference_frame_index,
        "note": (
            "MV-SAM3D fused canonical mesh. local_to_scene is the reference "
            "view's object-local to PyTorch3D-camera metric similarity."
        ),
        "objects": [
            {
                "index": 0,
                "mesh_obj": name,
                "local_to_scene": {
                    "translation": _as_list(translation),
                    "scale": _as_list(scale),
                    "quat_wxyz": _as_list(rotation),
                    "quat_xyzw": [float(v) for v in q_xyzw],
                },
            }
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layout.json").write_text(json.dumps(payload, indent=2) + "\n")


def _view_pose_record(
    *,
    view: int,
    frame_index: int,
    rotation,
    translation,
    scale,
    reference: bool,
    fit_iou: float | None,
) -> dict:
    """Build one metric pose record in both SAM3D and OpenCV camera axes."""

    from pytorch3d.transforms import quaternion_to_matrix

    quat = torch.as_tensor(_as_list(rotation), dtype=torch.float32)
    quat = quat / torch.linalg.norm(quat)
    rotation_p3d = quaternion_to_matrix(quat)
    translation_p3d = torch.as_tensor(_as_list(translation), dtype=torch.float32)
    p3d_to_opencv = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    canonical_z_up_to_y_up = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    object_to_camera = torch.eye(4, dtype=torch.float32)
    object_to_camera[:3, :3] = p3d_to_opencv @ rotation_p3d @ canonical_z_up_to_y_up
    object_to_camera[:3, 3] = p3d_to_opencv @ translation_p3d
    return {
        "view": view,
        "frame_index": frame_index,
        "scale": _as_list(scale),
        "rotation_wxyz_pytorch3d": _as_list(rotation),
        "translation_pytorch3d": _as_list(translation),
        "object_to_camera_opencv": object_to_camera.tolist(),
        "fit_iou": fit_iou,
        "reference": reference,
    }


def _save_view_poses(out_dir: Path, view_poses) -> None:
    """Persist the required per-view metric object-to-camera pose contract."""

    payload = {
        "schema_version": "2.0",
        "stage": "obj_recon",
        "coordinate_frame": "pytorch3d_camera",
        "pose_convention": "object_local_to_camera",
        "units": {"translation": "metres", "scale": "metres_per_mesh_unit"},
        "reference_view": 0,
        "reference_frame_index": view_poses[0]["frame_index"],
        "views": view_poses,
    }
    (out_dir / "view_poses.json").write_text(json.dumps(payload, indent=2) + "\n")


def reconstruct_multiview(
    masks_json: Path,
    *,
    geometry_json: Path,
    frames_json: Path | None = None,
    output_root: Path | None = None,
    prompt_ids: Sequence[str] | None = None,
    recon_args: MeshReconArgs,
    mv_args: MultiViewArgs,
    manual_frame_index: Sequence[int] | None = None,
) -> dict:
    """Run MV-SAM3D over selected keyframes; write fused meshes + reference poses."""
    masks_json = Path(masks_json).expanduser().resolve()
    if not masks_json.exists():
        raise FileNotFoundError(f"masks.json not found: {masks_json}")
    manifest = json.loads(masks_json.read_text(encoding="utf-8"))
    if manifest.get("stage") != "segment":
        raise ValueError(f"Not a segment manifest: {masks_json}")

    geometry_frames = load_geometry_manifest(Path(geometry_json))
    entries = manifest.get("entries", [])
    segment_ids = {int(e["index"]) for e in entries}
    usable = sorted(segment_ids & set(geometry_frames))
    if len(usable) < 2:
        raise ValueError(f"Need >=2 frames with mask+geometry, found {usable}")

    # Resolve video for the ffprobe strategy (manifest-driven if not overridden).
    video = mv_args.keyframe_video
    if video is None and frames_json is not None:
        fdata = json.loads(Path(frames_json).read_text(encoding="utf-8"))
        for key in ("source_video", "video", "source"):
            if fdata.get(key):
                video = Path(fdata[key])
                break

    keyframes = select_keyframes(
        usable,
        strategy=mv_args.keyframe_strategy,
        num_views=mv_args.num_views,
        max_views_cap=mv_args.max_views_cap,
        manual=manual_frame_index,
        video=video,
    )
    logger.info(
        "[mv] strategy={} -> keyframes={}", mv_args.keyframe_strategy, keyframes
    )

    # Stage dir mirrors the single-frame layout.
    segment_dir = masks_json.parent
    clip_root = segment_dir.parent
    if output_root is not None:
        stage_dir = (
            Path(output_root).expanduser().resolve() / clip_root.name / "obj_recon"
        )
    else:
        stage_dir = clip_root / "obj_recon"
    meshes_dir = stage_dir / "meshes" / "mv"
    stage_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Which object prompts to reconstruct (default: text prompts, not hands).
    available = [p["prompt_id"] for p in manifest.get("prompts", [])]
    if prompt_ids:
        selected = [p for p in prompt_ids if p in available]
    else:
        selected = [
            p["prompt_id"]
            for p in manifest.get("prompts", [])
            if p.get("input_type") == "text"
        ] or available
    if not selected:
        raise ValueError("No prompt ids selected for multi-view reconstruction.")

    masks_root = Path(manifest.get("masks_dir") or segment_dir / "masks")
    masks_root = (
        masks_root if masks_root.is_absolute() else (masks_json.parent / masks_root)
    )
    masks_root = masks_root.resolve()
    entry_by_idx = {int(e["index"]): e for e in entries}

    from obj_recon.device import resolve_torch_device, set_cuda_device_if_indexed

    device = resolve_torch_device(recon_args.device)
    set_cuda_device_if_indexed(device)
    pipeline = _load_mv_inference(recon_args)
    _patch_mv_compute_pointmap_intrinsics(pipeline)

    # Downsample factor so the model sees <= ~518px while we keep full-res masks.
    ds = 2

    results: list[dict] = []
    for prompt_id in selected:
        obj_name = _sanitize_prompt_id(prompt_id)
        images, vmasks, pointmaps, used_frames = [], [], [], []
        pipeline._mv_pending_intrinsics.clear()
        for idx in keyframes:
            entry = entry_by_idx[idx]
            # mask item for this prompt in this frame
            mask_path = None
            for pm in entry.get("prompt_masks", []) or []:
                if pm.get("prompt_id") == prompt_id and pm.get("has_mask"):
                    mask_path = masks_root / pm["mask_filename"]
                    break
            if mask_path is None or not mask_path.exists():
                logger.warning(
                    "[mv] frame {}: no mask for '{}', skipping view", idx, prompt_id
                )
                continue
            mask = load_mask(mask_path)
            if mask_is_empty(mask):
                logger.warning(
                    "[mv] frame {}: empty mask for '{}', skipping view", idx, prompt_id
                )
                continue
            rgba = load_rgba_image(
                resolve_frame_path(
                    entry, frames_json=frames_json, masks_json=masks_json
                )
            )
            rgba = _resize_rgba(rgba, ds)
            m_resized = np.asarray(
                Image.fromarray(mask_to_alpha(mask)).resize(
                    (rgba.shape[1], rgba.shape[0]), Image.NEAREST
                )
            )
            points = np.load(geometry_frames[idx].points_path)
            pts = _mv_pointmap(points, mask, downsample=ds)
            rgba = np.dstack([rgba[..., :3], m_resized])
            images.append(rgba)
            vmasks.append(None)  # mask already embedded in alpha channel
            pointmaps.append(pts)
            used_frames.append(idx)
            # Normalize against the original calibration resolution. Normalized
            # intrinsics are invariant to the image/point-map downsampling above.
            K = np.load(geometry_frames[idx].intrinsics_path)
            h, w = points.shape[:2]
            K = _normalize_intrinsics(K, w, h)
            pipeline._mv_pending_intrinsics.append(torch.from_numpy(K))

        if len(images) < 2:
            logger.warning("[mv] object '{}' has <2 usable views, skipping", prompt_id)
            continue

        weighting_config = None
        if mv_args.stage2_weighting:
            from sam3d_objects.utils.latent_weighting import WeightingConfig  # MV-only

            weighting_config = WeightingConfig(
                use_entropy=True,
                weight_source="entropy",
                entropy_alpha=mv_args.stage2_entropy_alpha,
                attention_layer=mv_args.stage2_attention_layer,
                attention_step=mv_args.stage2_attention_step,
                min_weight=mv_args.stage2_min_weight,
                weight_combine_mode="average",
                visibility_weight_ratio=0.5,
                visibility_callback=None,
            )

        logger.info(
            "[mv] Running MV-SAM3D for '{}' over frames {}", prompt_id, used_frames
        )
        n_queued = len(pipeline._mv_pending_intrinsics)
        with torch.no_grad():
            result = pipeline.run_multi_view(
                view_images=images,
                view_masks=vmasks,
                view_pointmaps=pointmaps,
                seed=mv_args.seed,
                mode=mv_args.mode,
                stage1_inference_steps=mv_args.stage1_inference_steps,
                stage2_inference_steps=mv_args.stage2_inference_steps,
                with_mesh_postprocess=recon_args.with_mesh_postprocess,
                with_texture_baking=recon_args.with_texture_baking,
                use_vertex_color=recon_args.use_vertex_color,
                ss_weighting=mv_args.ss_weighting,
                ss_entropy_layer=mv_args.ss_entropy_layer,
                ss_entropy_alpha=mv_args.ss_entropy_alpha,
                ss_warmup_steps=1,
                weighting_config=weighting_config,
            )
        leftover = len(pipeline._mv_pending_intrinsics)
        if leftover != n_queued - len(images):
            logger.warning(
                "[mv] intrinsics queue drift: queued {} views, {} leftover (expected {})",
                n_queued,
                leftover,
                n_queued - len(images),
            )
        pipeline._mv_pending_intrinsics.clear()

        mesh = result.get("glb")
        if mesh is None:
            logger.warning("[mv] no mesh produced for '{}'", prompt_id)
            continue

        decoded_view_poses = result.get("all_view_poses_decoded") or []
        if len(decoded_view_poses) != len(used_frames):
            raise RuntimeError(
                "MV-SAM3D must establish one pose per used view: "
                f"got {len(decoded_view_poses)} poses for {len(used_frames)} views."
            )
        if pipeline.layout_post_optimization_method is None:
            raise RuntimeError(
                "Per-view metric pose fitting is required for MV output."
            )

        view_poses = []
        for view, (frame_index, decoded) in enumerate(
            zip(used_frames, decoded_view_poses, strict=True)
        ):
            try:
                points = np.load(geometry_frames[frame_index].points_path)
                h, w = points.shape[:2]
                intrinsics = torch.from_numpy(
                    _normalize_intrinsics(
                        np.load(geometry_frames[frame_index].intrinsics_path), w, h
                    )
                ).to(pipeline.device)
                pose_input = {
                    key: torch.as_tensor(decoded[key], device=pipeline.device)
                    for key in ("rotation", "translation", "scale")
                }
                fitted = pipeline.run_post_optimization(
                    deepcopy(mesh),
                    intrinsics,
                    pose_input,
                    result["view_ss_input_dicts"][view],
                )
                logger.info(
                    "[mv] view {} frame {} pose fit: iou={}",
                    view,
                    frame_index,
                    fitted.get("iou"),
                )
            except Exception:
                logger.exception(
                    "[mv] metric pose fitting failed for view {} frame {}",
                    view,
                    frame_index,
                )
                raise
            fit_iou = float(fitted.get("iou", -1.0))
            accepted_fit = fit_iou >= mv_args.min_fit_iou
            pose = fitted if accepted_fit else decoded
            if not accepted_fit:
                logger.warning(
                    "[mv] view {} frame {} rejected pose fit (iou={:.4f}); using decoded pose",
                    view,
                    frame_index,
                    fit_iou,
                )
            record = _view_pose_record(
                view=view,
                frame_index=frame_index,
                rotation=pose["rotation"],
                translation=pose["translation"],
                scale=pose["scale"],
                reference=view == 0,
                fit_iou=fit_iou,
            )
            record["fit_accepted"] = accepted_fit
            view_poses.append(record)

        out_dir = meshes_dir / obj_name
        out_dir.mkdir(parents=True, exist_ok=True)
        glb_path = out_dir / f"{obj_name}.glb"
        obj_path = out_dir / f"{obj_name}.obj"
        mesh.export(str(glb_path))
        mesh.export(str(obj_path))
        logger.info("[mv] saved fused textured mesh: {}", glb_path)

        reference_pose = view_poses[0]
        canonical_scale = reference_pose["scale"]
        for view_pose in view_poses:
            predicted_scale = np.asarray(view_pose["scale"], dtype=np.float64)
            reference_scale = np.asarray(canonical_scale, dtype=np.float64)
            scale_ratio = predicted_scale / reference_scale
            view_pose["predicted_scale"] = view_pose["scale"]
            view_pose["predicted_scale_ratio"] = scale_ratio.tolist()
            view_pose["scale"] = canonical_scale
        _save_layout(
            out_dir,
            obj_path.name,
            reference_pose["rotation_wxyz_pytorch3d"],
            reference_pose["translation_pytorch3d"],
            reference_pose["scale"],
            reference_frame_index=used_frames[0],
        )
        _save_view_poses(out_dir, view_poses)
        scale = reference_pose["scale"]

        results.append(
            {
                "prompt_id": prompt_id,
                "object": obj_name,
                "keyframes": used_frames,
                "reference_frame_index": used_frames[0],
                "glb": str(glb_path),
                "obj": str(obj_path),
                "layout": str(out_dir / "layout.json"),
                "view_poses": str(out_dir / "view_poses.json"),
                "scale": _as_list(scale),
            }
        )

    _write_manifest(
        stage_dir,
        meshes_dir,
        masks_json,
        geometry_json,
        frames_json,
        recon_args,
        mv_args,
        keyframes,
        selected,
        results,
    )
    logger.info("[mv] Done: {} objects reconstructed -> {}", len(results), meshes_dir)
    return {"meshes_dir": str(meshes_dir), "objects": results, "keyframes": keyframes}


def _write_manifest(
    stage_dir,
    meshes_dir,
    masks_json,
    geometry_json,
    frames_json,
    recon_args,
    mv_args,
    keyframes,
    prompts,
    results,
) -> None:
    (stage_dir / "mv_meshes.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "stage": "obj_recon",
                "mode": "multi_view",
                "source_masks_json": str(masks_json),
                "source_geometry_json": str(geometry_json),
                "source_frames_json": str(frames_json) if frames_json else None,
                "keyframes": keyframes,
                "prompts": list(prompts),
                "meshes_dir": str(meshes_dir),
                "objects": results,
            },
            indent=2,
        )
        + "\n"
    )
    (stage_dir / "mv_config.json").write_text(
        json.dumps(
            {
                "package": {
                    "name": "obj_recon",
                    "version": __version__,
                    "mode": "multi_view",
                },
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "multiview": {
                    "keyframe_strategy": mv_args.keyframe_strategy,
                    "num_views": mv_args.num_views,
                    "max_views_cap": mv_args.max_views_cap,
                    "mode": mv_args.mode,
                    "ss_weighting": mv_args.ss_weighting,
                    "stage2_weighting": mv_args.stage2_weighting,
                    "stage2_entropy_alpha": mv_args.stage2_entropy_alpha,
                },
                "reconstruction": {
                    "config": str(recon_args.config.expanduser().resolve()),
                    "device": recon_args.device,
                    "with_texture_baking": recon_args.with_texture_baking,
                },
            },
            indent=2,
        )
        + "\n"
    )
