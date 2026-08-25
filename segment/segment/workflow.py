"""Video-mode frame-by-frame text-prompt segmentation.

The SAM3 model and each frame's image embedding are reused across all configured
text prompts. Outputs include an aggregate mask, one mask per prompt, a JSON
manifest that preserves prompt identity, and an optional legend-bearing overlay.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

from segment import __version__
from segment.anchors import AnchorManifest, load_anchor_manifest
from segment.frames import FrameManifest, load_frame_manifest
from segment.media import load_rgb_image, mask_stats, save_mask, save_prompt_overlay
from segment.sam_mask import (
    DEFAULT_TEXT_PROMPTS,
    PromptMaskResult,
    Sam3MaskGenerator,
    SamMaskArgs,
    generate_prompt_masks,
    prompt_colors_rgb,
    resolve_text_prompts,
    union_prompt_masks,
)

MASK_FILENAME_PATTERN = "{:06d}.png"
VIS_FILENAME_PATTERN = "{:06d}.jpg"


@dataclass
class SegmentVideoArgs:
    """Arguments for frame-by-frame segmentation of a whole video."""

    frames_json: Path
    """Path to the ``process`` stage's ``frames.json`` (the frame manifest)."""

    output_root: Path = Path(__file__).parents[2] / "outputs"
    """Root under which ``<clip_stem>/segment/`` is created."""

    vis: bool = True
    """Write an original frame + mask overlay image for every processed frame."""

    max_frames: int | None = None
    """Limit the number of frames processed (None = all frames in the manifest)."""

    anchors_json: Path | None = None
    """Generic per-frame point/box anchor manifest for geometric prompts."""

    sam_mask: SamMaskArgs = field(default_factory=SamMaskArgs)
    """SAM3 segmentation settings (checkpoint, prompts, thresholds, ...)."""


@dataclass(frozen=True)
class PromptMaskEntry:
    """One prompt's mask record for a frame."""

    prompt_id: str
    text_prompt: str | None
    input_type: str
    anchor: dict | None
    mask_filename: str
    has_mask: bool
    instance_count: int
    area: int
    bbox: dict | None

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "text_prompt": self.text_prompt,
            "input_type": self.input_type,
            "anchor": self.anchor,
            "mask_filename": self.mask_filename,
            "has_mask": self.has_mask,
            "instance_count": self.instance_count,
            "area": self.area,
            "bbox": self.bbox,
        }


@dataclass(frozen=True)
class MaskEntry:
    """Per-frame mask record written into ``masks.json``."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    mask_filename: str | None
    vis_filename: str | None
    has_mask: bool
    instance_count: int
    area: int
    bbox: dict | None
    prompt_masks: tuple[PromptMaskEntry, ...] = ()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frame_filename": self.frame_filename,
            "timestamp_sec": self.timestamp_sec,
            "mask_filename": self.mask_filename,
            "vis_filename": self.vis_filename,
            "has_mask": self.has_mask,
            "instance_count": self.instance_count,
            "area": self.area,
            "bbox": self.bbox,
            "prompt_masks": [item.to_dict() for item in self.prompt_masks],
        }


@dataclass(frozen=True)
class SegmentVideoOutputs:
    """Everything produced by one video segmentation run."""

    clip_root: Path
    stage_dir: Path
    masks_dir: Path
    masks_vis_dir: Path | None
    masks_json_path: Path
    config_json_path: Path
    entries: list[MaskEntry]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_existing_entries(masks_json: Path) -> dict[int, MaskEntry]:
    """Reuse previously written mask entries (idempotent non-overwrite runs)."""

    if not masks_json.exists():
        return {}
    try:
        data = json.loads(masks_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    existing: dict[int, MaskEntry] = {}
    for raw in data.get("entries", []):
        try:
            prompt_masks = tuple(
                PromptMaskEntry(
                    prompt_id=item["prompt_id"],
                    text_prompt=item.get("text_prompt"),
                    input_type=item.get("input_type", "text"),
                    anchor=item.get("anchor"),
                    mask_filename=item["mask_filename"],
                    has_mask=bool(item.get("has_mask")),
                    instance_count=int(item.get("instance_count", 0)),
                    area=int(item.get("area", 0)),
                    bbox=item.get("bbox"),
                )
                for item in raw.get("prompt_masks", [])
            )
            existing[int(raw["index"])] = MaskEntry(
                index=int(raw["index"]),
                frame_filename=raw["frame_filename"],
                timestamp_sec=raw.get("timestamp_sec"),
                mask_filename=raw.get("mask_filename"),
                vis_filename=raw.get("vis_filename"),
                has_mask=bool(raw.get("has_mask")),
                instance_count=int(raw.get("instance_count", 0)),
                area=int(raw.get("area", 0)),
                bbox=raw.get("bbox"),
                prompt_masks=prompt_masks,
            )
        except (KeyError, TypeError, ValueError):
            continue
    return existing


def _prompt_manifest(
    prompts: tuple[str, ...],
    colors: tuple[tuple[int, int, int], ...],
    masks_dir: Path,
    anchor_types: dict[str, str],
) -> list[dict]:
    return [
        {
            "prompt_id": text_prompt,
            "text_prompt": None if text_prompt in anchor_types else text_prompt,
            "input_type": (
                f"anchor_{anchor_types[text_prompt]}"
                if text_prompt in anchor_types
                else "text"
            ),
            "anchor_source": "json" if text_prompt in anchor_types else None,
            "color_rgb": list(color),
            "masks_dir": str(masks_dir / text_prompt),
        }
        for text_prompt, color in zip(prompts, colors)
    ]


def _mask_manifest_dict(
    manifest: FrameManifest,
    args: SegmentVideoArgs,
    masks_dir: Path,
    masks_vis_dir: Path | None,
    entries: list[MaskEntry],
    prompts: tuple[str, ...],
    colors: tuple[tuple[int, int, int], ...],
    anchor_types: dict[str, str],
    anchors_json_path: Path | None,
) -> dict:
    return {
        "schema_version": "1.2",
        "stage": "segment",
        "source_frames_json": str(args.frames_json.expanduser().resolve()),
        "source_video": manifest.source_video,
        "fps": manifest.fps,
        "width": manifest.width,
        "height": manifest.height,
        "frame_format": manifest.format,
        "frame_count": manifest.frame_count,
        "processed_count": len(entries),
        "masks_dir": str(masks_dir),
        "masks_vis_dir": str(masks_vis_dir) if masks_vis_dir is not None else None,
        "mask_format": "png",
        "vis_enabled": args.vis,
        "anchor_source": str(anchors_json_path) if anchors_json_path else None,
        "prompts": _prompt_manifest(prompts, colors, masks_dir, anchor_types),
        "entries": [entry.to_dict() for entry in entries],
    }


def _config_dict(
    args: SegmentVideoArgs,
    manifest: FrameManifest,
    prompts: tuple[str, ...],
    colors: tuple[tuple[int, int, int], ...],
    anchor_types: dict[str, str],
    anchors_json_path: Path | None,
) -> dict:
    checkpoint = (
        str(args.sam_mask.checkpoint.expanduser())
        if args.sam_mask.checkpoint is not None
        else None
    )
    return {
        "package": {"name": "segment", "version": __version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "frames_json": str(args.frames_json.expanduser().resolve()),
            "source_video": manifest.source_video,
            "fps": manifest.fps,
            "width": manifest.width,
            "height": manifest.height,
            "frame_format": manifest.format,
            "frame_count": manifest.frame_count,
        },
        "segment": {
            "checkpoint": checkpoint,
            "allow_hf_download": args.sam_mask.allow_hf_download,
            "device": args.sam_mask.device,
            "text_prompts": [
                prompt for prompt in prompts if prompt not in anchor_types
            ],
            "anchors_json": str(anchors_json_path) if anchors_json_path else None,
            "score_threshold": args.sam_mask.score_threshold,
            "overlay_alpha": args.sam_mask.overlay_alpha,
            "prompt_colors_rgb": [list(color) for color in colors],
            "vis": args.vis,
            "overwrite": args.sam_mask.overwrite,
            "max_frames": args.max_frames,
        },
        "software": {},
    }


def _can_reuse_entry(
    prior: MaskEntry | None,
    prompts: tuple[str, ...],
    anchor_types: dict[str, str],
    mask_path: Path,
    vis_path: Path | None,
    masks_dir: Path,
) -> bool:
    if (
        prior is None
        or not mask_path.exists()
        or (vis_path is not None and not vis_path.exists())
    ):
        return False
    if tuple(item.prompt_id for item in prior.prompt_masks) != prompts:
        return False
    for item in prior.prompt_masks:
        expected_type = (
            f"anchor_{anchor_types[item.prompt_id]}"
            if item.prompt_id in anchor_types
            else "text"
        )
        if item.input_type != expected_type:
            return False
    return all((masks_dir / item.mask_filename).exists() for item in prior.prompt_masks)


def run_video_segment(
    args: SegmentVideoArgs,
    generator: Sam3MaskGenerator | None = None,
) -> SegmentVideoOutputs:
    """Segment every frame once for all configured text prompts."""

    if args.max_frames is not None and args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")

    configured_text_prompts = resolve_text_prompts(args.sam_mask)
    frames_json = args.frames_json.expanduser().resolve()
    manifest = load_frame_manifest(frames_json)

    anchor_manifest: AnchorManifest | None = None
    if args.anchors_json is not None:
        anchor_manifest = load_anchor_manifest(args.anchors_json)
        if (
            args.sam_mask.text_prompt is None
            and tuple(args.sam_mask.text_prompts) == DEFAULT_TEXT_PROMPTS
        ):
            configured_text_prompts = ()
    anchor_prompt_order = (
        tuple(target.prompt_id for target in anchor_manifest.targets)
        if anchor_manifest is not None
        else ()
    )
    anchor_types = (
        {target.prompt_id: target.anchor_type for target in anchor_manifest.targets}
        if anchor_manifest is not None
        else {}
    )
    anchors = anchor_manifest.anchors if anchor_manifest is not None else {}

    prompts = anchor_prompt_order + tuple(
        prompt
        for prompt in configured_text_prompts
        if prompt not in anchor_prompt_order
    )
    if len(prompts) > 8:
        raise ValueError("At most 8 combined text and anchor prompts are supported.")
    colors = prompt_colors_rgb(args.sam_mask, len(prompts))

    clip_stem = frames_json.parent.parent.name
    clip_root = args.output_root.expanduser().resolve() / clip_stem
    stage_dir = clip_root / "segment"
    masks_dir = stage_dir / "masks"
    masks_vis_dir = stage_dir / "masks_vis" if args.vis else None

    if args.sam_mask.overwrite:
        if masks_dir.exists():
            shutil.rmtree(masks_dir)
        if masks_vis_dir is not None and masks_vis_dir.exists():
            shutil.rmtree(masks_vis_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for text_prompt in prompts:
        (masks_dir / text_prompt).mkdir(parents=True, exist_ok=True)
    if masks_vis_dir is not None:
        masks_vis_dir.mkdir(parents=True, exist_ok=True)

    selected = manifest.entries
    if args.max_frames is not None:
        selected = selected[: args.max_frames]

    active_generator = generator or Sam3MaskGenerator(
        args.sam_mask, enable_anchor_prompts=bool(anchor_types)
    )
    existing = _load_existing_entries(stage_dir / "masks.json")

    entries: list[MaskEntry] = []
    for frame in selected:
        mask_filename = MASK_FILENAME_PATTERN.format(frame.index)
        vis_filename = (
            VIS_FILENAME_PATTERN.format(frame.index)
            if masks_vis_dir is not None
            else None
        )
        mask_path = masks_dir / mask_filename
        vis_path = (
            masks_vis_dir / vis_filename
            if masks_vis_dir is not None and vis_filename is not None
            else None
        )

        prior = existing.get(frame.index)
        if not args.sam_mask.overwrite and _can_reuse_entry(
            prior,
            prompts,
            anchor_types,
            mask_path,
            vis_path,
            masks_dir,
        ):
            entries.append(prior)
            continue

        frame_rgb = load_rgb_image(frame.path)
        text_prompts = tuple(prompt for prompt in prompts if prompt not in anchor_types)
        results_by_prompt = {}
        if text_prompts:
            results_by_prompt = {
                result.text_prompt: result
                for result in generate_prompt_masks(
                    active_generator, frame_rgb, text_prompts
                )
            }
        frame_anchors = [
            (prompt, anchors[prompt][frame.index], anchor_types[prompt])
            for prompt in prompts
            if prompt in anchor_types and frame.index in anchors[prompt]
        ]
        if frame_anchors:
            if not hasattr(active_generator, "segment_anchors"):
                raise TypeError(
                    "The supplied generator does not support geometric anchors."
                )
            for result in active_generator.segment_anchors(frame_rgb, frame_anchors):
                results_by_prompt[result.text_prompt] = result

        prompt_results = []
        for prompt in prompts:
            result = results_by_prompt.get(prompt)
            if result is None:
                result = PromptMaskResult(
                    prompt,
                    np.zeros(frame_rgb.shape[:2], dtype=np.uint8),
                    0,
                )
            prompt_results.append(result)
        aggregate_mask = union_prompt_masks(prompt_results, frame_rgb.shape[:2])
        aggregate_stats = mask_stats(aggregate_mask)

        save_mask(aggregate_mask, mask_path, args.sam_mask.overwrite)
        prompt_entries: list[PromptMaskEntry] = []
        for result in prompt_results:
            prompt_id = result.text_prompt
            relative_mask_filename = f"{prompt_id}/{mask_filename}"
            save_mask(
                result.mask,
                masks_dir / relative_mask_filename,
                args.sam_mask.overwrite,
            )
            stats = mask_stats(result.mask)
            is_anchor = prompt_id in anchor_types
            frame_anchor = anchors[prompt_id].get(frame.index) if is_anchor else None
            anchor_dict = None
            if frame_anchor is not None:
                anchor_dict = {
                    "point_xy": (
                        list(frame_anchor.point_xy)
                        if frame_anchor.point_xy is not None
                        else None
                    ),
                    "box_xyxy": (
                        list(frame_anchor.box_xyxy)
                        if frame_anchor.box_xyxy is not None
                        else None
                    ),
                    "confidence": frame_anchor.confidence,
                }
            prompt_entries.append(
                PromptMaskEntry(
                    prompt_id=prompt_id,
                    text_prompt=None if is_anchor else result.text_prompt,
                    input_type=(
                        f"anchor_{anchor_types[prompt_id]}" if is_anchor else "text"
                    ),
                    anchor=anchor_dict,
                    mask_filename=relative_mask_filename,
                    has_mask=stats.has_mask,
                    instance_count=result.instance_count,
                    area=stats.area,
                    bbox=stats.to_dict()["bbox"],
                )
            )

        if vis_path is not None:
            save_prompt_overlay(
                frame_rgb,
                [result.mask for result in prompt_results],
                prompts,
                vis_path,
                alpha=args.sam_mask.overlay_alpha,
                colors_rgb=colors,
                overwrite=args.sam_mask.overwrite,
            )

        entries.append(
            MaskEntry(
                index=frame.index,
                frame_filename=frame.frame_filename,
                timestamp_sec=frame.timestamp_sec,
                mask_filename=mask_filename,
                vis_filename=vis_filename,
                has_mask=aggregate_stats.has_mask,
                instance_count=sum(result.instance_count for result in prompt_results),
                area=aggregate_stats.area,
                bbox=aggregate_stats.to_dict()["bbox"],
                prompt_masks=tuple(prompt_entries),
            )
        )

    _write_json(
        stage_dir / "masks.json",
        _mask_manifest_dict(
            manifest,
            args,
            masks_dir,
            masks_vis_dir,
            entries,
            prompts,
            colors,
            anchor_types,
            anchor_manifest.source_path if anchor_manifest else None,
        ),
    )
    _write_json(
        stage_dir / "config.json",
        _config_dict(
            args,
            manifest,
            prompts,
            colors,
            anchor_types,
            anchor_manifest.source_path if anchor_manifest else None,
        ),
    )

    with_mask = sum(1 for entry in entries if entry.has_mask)
    logger.info(
        "[segment] Done: processed={} prompts={} has_mask={} vis={} out={}",
        len(entries),
        len(prompts),
        with_mask,
        args.vis,
        stage_dir,
    )

    return SegmentVideoOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        masks_dir=masks_dir,
        masks_vis_dir=masks_vis_dir,
        masks_json_path=stage_dir / "masks.json",
        config_json_path=stage_dir / "config.json",
        entries=entries,
    )
