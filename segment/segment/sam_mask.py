"""SAM3 text-prompt mask segmentation for a single RGB image."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from loguru import logger
from PIL import Image

from segment import (
    load_rgb_image,
    resolve_torch_device,
    save_mask,
    set_cuda_device_if_indexed,
)
from segment.anchors import Anchor
from segment.media import PROMPT_COLORS_RGB, save_prompt_overlay

DEFAULT_TEXT_PROMPT = "human hand and arm"
DEFAULT_TEXT_PROMPTS = (DEFAULT_TEXT_PROMPT,)
DEFAULT_MASK_COLOR_RGB = (0, 0, 255)


@dataclass
class SamMaskArgs:
    """Arguments for SAM3 mask generation on an image."""

    checkpoint: Path | None = Path(__file__).parents[2] / "ckpts/sam3/sam3.pt"
    allow_hf_download: bool = False
    device: str = "auto"
    text_prompts: tuple[str, ...] = DEFAULT_TEXT_PROMPTS
    """One or more text prompts, evaluated against one shared image embedding."""
    text_prompt: str | None = None
    """Deprecated single-prompt alias. Use ``text_prompts`` for new calls."""
    score_threshold: float = 0.1
    overlay_alpha: float = 0.5
    mask_color_rgb: tuple[int, int, int] = DEFAULT_MASK_COLOR_RGB
    """Color for the first prompt; later prompts use the fixed categorical palette."""
    overwrite: bool = True


@dataclass
class SamMaskCliArgs:
    """CLI wrapper for SAM3 mask generation on a single image."""

    image_path: Path
    output_dir: Path
    sam_mask: SamMaskArgs = field(default_factory=SamMaskArgs)


@dataclass(frozen=True)
class PromptMaskResult:
    """Union mask and instance count for one text prompt."""

    text_prompt: str
    mask: np.ndarray
    instance_count: int


@dataclass(frozen=True)
class SamMaskOutputs:
    """Output paths produced by SAM3 mask generation."""

    mask_path: Path
    overlay_path: Path
    mask: np.ndarray
    prompt_mask_paths: tuple[Path, ...]
    prompt_results: tuple[PromptMaskResult, ...]


@dataclass(frozen=True)
class SamMaskInstance:
    """Single SAM3 mask instance returned for a text prompt."""

    mask: np.ndarray
    score: float


def resolve_text_prompts(args: SamMaskArgs) -> tuple[str, ...]:
    """Resolve the multi-prompt option and the legacy single-prompt alias."""

    configured = tuple(args.text_prompts)
    if args.text_prompt is not None:
        if configured != DEFAULT_TEXT_PROMPTS:
            raise ValueError("Pass either --text-prompts or --text-prompt, not both.")
        configured = (args.text_prompt,)

    prompts = tuple(prompt.strip() for prompt in configured)
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("At least one non-empty mask prompt is required.")
    if len(set(prompts)) != len(prompts):
        raise ValueError("Mask prompts must be unique.")
    invalid_paths = [
        prompt
        for prompt in prompts
        if prompt in {".", ".."} or Path(prompt).name != prompt
    ]
    if invalid_paths:
        raise ValueError(
            "Mask prompts are used as output directory names and cannot contain "
            f"path separators: {invalid_paths}"
        )
    if len(prompts) > len(PROMPT_COLORS_RGB):
        raise ValueError(
            f"At most {len(PROMPT_COLORS_RGB)} mask prompts are supported."
        )
    return prompts


def prompt_colors_rgb(
    args: SamMaskArgs, prompt_count: int
) -> tuple[tuple[int, int, int], ...]:
    """Return stable colors for prompts in their configured order."""

    if prompt_count > len(PROMPT_COLORS_RGB):
        raise ValueError(
            f"At most {len(PROMPT_COLORS_RGB)} mask prompts are supported."
        )
    if len(args.mask_color_rgb) != 3 or any(
        value < 0 or value > 255 for value in args.mask_color_rgb
    ):
        raise ValueError("--mask-color-rgb values must be within [0, 255].")

    colors = list(PROMPT_COLORS_RGB[:prompt_count])
    if colors:
        colors[0] = tuple(args.mask_color_rgb)
    return tuple(colors)


def _score_to_float(score: Any) -> float:
    if torch.is_tensor(score):
        return float(score.detach().cpu().item())
    return float(score)


def _mask_to_numpy(mask: Any) -> np.ndarray:
    if torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    while mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np, axis=0)
    return mask_np


def _instances_from_output(
    output: dict,
    expected_shape: tuple[int, int],
    threshold: float,
) -> list[SamMaskInstance]:
    instances: list[SamMaskInstance] = []
    for raw_mask, raw_score in zip(output["masks"], output["scores"]):
        score = _score_to_float(raw_score)
        if score < threshold:
            continue
        mask_np = _mask_to_numpy(raw_mask)
        if mask_np.shape != expected_shape:
            raise ValueError(
                f"SAM3 mask shape mismatch: expected {expected_shape}, got {mask_np.shape}"
            )
        instances.append(
            SamMaskInstance(
                mask=(mask_np > 0).astype(np.uint8),
                score=score,
            )
        )
    return instances


def _union_instances(
    instances: Sequence[SamMaskInstance], shape: tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for instance in instances:
        mask[instance.mask > 0] = 255
    return mask


def union_prompt_masks(
    prompt_results: Sequence[PromptMaskResult], shape: tuple[int, int]
) -> np.ndarray:
    """Union all prompt masks into the legacy aggregate binary mask."""

    mask = np.zeros(shape, dtype=np.uint8)
    for result in prompt_results:
        mask[result.mask > 0] = 255
    return mask


class Sam3MaskGenerator:
    """Reusable SAM3 image segmenter."""

    def __init__(self, args: SamMaskArgs, enable_anchor_prompts: bool = False) -> None:
        if not 0 <= args.score_threshold <= 1:
            raise ValueError("--score-threshold must be within [0, 1].")

        checkpoint = args.checkpoint.expanduser() if args.checkpoint else None
        if checkpoint is None and not args.allow_hf_download:
            raise ValueError(
                "Pass --sam-mask.checkpoint or --sam-mask.allow-hf-download "
                "to let SAM3 download weights from Hugging Face."
            )
        if checkpoint is not None and not checkpoint.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

        self.device = resolve_torch_device(args.device)
        self.score_threshold = args.score_threshold
        set_cuda_device_if_indexed(self.device)

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        logger.info(
            "[SAM3] Loading model: device={}, source={}",
            self.device,
            checkpoint or "Hugging Face",
        )
        build_device = str(self.device) if self.device.type == "cuda" else "cpu"
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint) if checkpoint else None,
            load_from_HF=checkpoint is None,
            device=build_device,
            enable_inst_interactivity=enable_anchor_prompts,
        )
        model.to(self.device)

        self.processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=args.score_threshold,
        )
        self.model = model
        self.anchor_predictor = model.inst_interactive_predictor

    def _inference_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def segment_prompt_instances(
        self,
        frame_rgb: np.ndarray,
        text_prompts: Sequence[str],
        score_threshold: float | None = None,
    ) -> list[list[SamMaskInstance]]:
        """Infer several prompts while computing the image embedding only once."""

        prompts = tuple(text_prompts)
        if not prompts:
            raise ValueError("At least one mask prompt is required.")

        threshold = self.score_threshold if score_threshold is None else score_threshold
        if not 0 <= threshold <= 1:
            raise ValueError("score_threshold must be within [0, 1].")

        previous_threshold = getattr(self.processor, "confidence_threshold", None)
        if hasattr(self.processor, "set_confidence_threshold"):
            self.processor.set_confidence_threshold(threshold)

        image = Image.fromarray(frame_rgb)
        expected_shape = frame_rgb.shape[:2]
        instance_groups: list[list[SamMaskInstance]] = []
        try:
            with self._inference_context():
                inference_state = self.processor.set_image(image)
                for text_prompt in prompts:
                    output = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=text_prompt,
                    )
                    # Sam3Processor returns and mutates the same state dict for
                    # every prompt, so materialize each result before the next call.
                    instance_groups.append(
                        _instances_from_output(output, expected_shape, threshold)
                    )
        finally:
            if (
                previous_threshold is not None
                and previous_threshold != threshold
                and hasattr(self.processor, "set_confidence_threshold")
            ):
                self.processor.set_confidence_threshold(previous_threshold)

        return instance_groups

    def segment_instances(
        self,
        frame_rgb: np.ndarray,
        text_prompt: str,
        score_threshold: float | None = None,
    ) -> list[SamMaskInstance]:
        return self.segment_prompt_instances(
            frame_rgb,
            [text_prompt],
            score_threshold,
        )[0]

    def segment_prompts(
        self,
        frame_rgb: np.ndarray,
        text_prompts: Sequence[str],
    ) -> list[PromptMaskResult]:
        prompts = tuple(text_prompts)
        instance_groups = self.segment_prompt_instances(frame_rgb, prompts)
        return [
            PromptMaskResult(
                text_prompt=text_prompt,
                mask=_union_instances(instances, frame_rgb.shape[:2]),
                instance_count=len(instances),
            )
            for text_prompt, instances in zip(prompts, instance_groups)
        ]

    def segment_anchors(
        self,
        frame_rgb: np.ndarray,
        anchors: Sequence[tuple[str, Anchor, str]],
    ) -> list[PromptMaskResult]:
        """Segment several objects from HaWoR anchors with one image embedding."""

        if self.anchor_predictor is None:
            raise RuntimeError(
                "SAM3 anchor prompting was not enabled at model load time."
            )
        with self._inference_context():
            inference_state = self.processor.set_image(Image.fromarray(frame_rgb))
        results: list[PromptMaskResult] = []
        for prompt_id, anchor, anchor_type in anchors:
            if anchor_type == "point":
                if anchor.point_xy is None:
                    raise ValueError(f"Missing point anchor for {prompt_id!r}.")
                masks, scores, _ = self.model.predict_inst(
                    inference_state,
                    point_coords=np.asarray([anchor.point_xy], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int32),
                    multimask_output=True,
                )
            elif anchor_type == "box":
                if anchor.box_xyxy is None:
                    raise ValueError(f"Missing box anchor for {prompt_id!r}.")
                masks, scores, _ = self.model.predict_inst(
                    inference_state,
                    box=np.asarray(anchor.box_xyxy, dtype=np.float32),
                    multimask_output=False,
                )
            else:
                raise ValueError(f"Unsupported anchor type: {anchor_type}")

            if len(masks) == 0:
                mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
                instance_count = 0
            else:
                best_index = int(np.argmax(np.asarray(scores)))
                mask = (np.asarray(masks[best_index]) > 0).astype(np.uint8) * 255
                instance_count = 1
            results.append(PromptMaskResult(prompt_id, mask, instance_count))
        return results

    def segment_anchor(
        self,
        frame_rgb: np.ndarray,
        text_prompt: str,
        anchor: Anchor,
        anchor_type: str,
    ) -> PromptMaskResult:
        return self.segment_anchors(frame_rgb, [(text_prompt, anchor, anchor_type)])[0]

    def segment(
        self, frame_rgb: np.ndarray, text_prompt: str
    ) -> tuple[np.ndarray, int]:
        result = self.segment_prompts(frame_rgb, [text_prompt])[0]
        return result.mask, result.instance_count


def generate_prompt_masks(
    generator: Any,
    frame_rgb: np.ndarray,
    text_prompts: Sequence[str],
) -> list[PromptMaskResult]:
    """Call the multi-prompt API, with a legacy generator fallback for tests/users."""

    prompts = tuple(text_prompts)
    if hasattr(generator, "segment_prompts"):
        return list(generator.segment_prompts(frame_rgb, prompts))

    results: list[PromptMaskResult] = []
    for text_prompt in prompts:
        mask, instance_count = generator.segment(frame_rgb, text_prompt)
        results.append(
            PromptMaskResult(
                text_prompt=text_prompt,
                mask=mask,
                instance_count=instance_count,
            )
        )
    return results


def process_sam_mask(
    image_path: Path,
    output_dir: Path,
    args: SamMaskArgs,
    generator: Sam3MaskGenerator | None = None,
) -> SamMaskOutputs:
    """Run SAM3 mask generation for one image and one or more prompts."""

    image_path = image_path.expanduser()
    prompts = resolve_text_prompts(args)
    colors = prompt_colors_rgb(args, len(prompts))

    mask_path = output_dir / "hand_seg.png"
    overlay_path = output_dir / "hand_seg_vis.jpg"

    frame_rgb = load_rgb_image(image_path)
    active_generator = generator or Sam3MaskGenerator(args)
    prompt_results = generate_prompt_masks(active_generator, frame_rgb, prompts)
    mask = union_prompt_masks(prompt_results, frame_rgb.shape[:2])

    save_mask(mask, mask_path, args.overwrite)
    prompt_mask_paths: list[Path] = []
    for result in prompt_results:
        prompt_mask_path = output_dir / "masks" / result.text_prompt / "mask.png"
        save_mask(result.mask, prompt_mask_path, args.overwrite)
        prompt_mask_paths.append(prompt_mask_path)

    save_prompt_overlay(
        frame_rgb,
        [result.mask for result in prompt_results],
        prompts,
        overlay_path,
        alpha=args.overlay_alpha,
        colors_rgb=colors,
        overwrite=args.overwrite,
    )

    return SamMaskOutputs(
        mask_path=mask_path,
        overlay_path=overlay_path,
        mask=mask,
        prompt_mask_paths=tuple(prompt_mask_paths),
        prompt_results=tuple(prompt_results),
    )


if __name__ == "__main__":
    args = tyro.cli(SamMaskCliArgs)
    process_sam_mask(
        image_path=args.image_path,
        output_dir=args.output_dir,
        args=args.sam_mask,
    )
