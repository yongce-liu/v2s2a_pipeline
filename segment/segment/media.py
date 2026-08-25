"""Image and mask helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Fixed categorical order. The first eight colors pass the project palette checks;
# prompts beyond eight are rejected rather than silently reusing a color.
PROMPT_COLORS_RGB: tuple[tuple[int, int, int], ...] = (
    (0, 0, 255),
    (235, 104, 52),
    (27, 175, 122),
    (237, 161, 0),
    (232, 123, 164),
    (0, 131, 0),
    (74, 58, 167),
    (227, 73, 72),
)


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_is_empty(mask: np.ndarray) -> bool:
    return not np.any(mask > 0)


@dataclass(frozen=True)
class MaskStats:
    """Geometry summary of one binary mask (row/col pixel coordinates)."""

    has_mask: bool
    area: int
    """Number of foreground pixels."""
    bbox: tuple[int, int, int, int] | None
    """Inclusive bounding box ``(min_row, min_col, max_row, max_col)`` or None."""

    def to_dict(self) -> dict:
        bbox = None
        if self.bbox is not None:
            min_row, min_col, max_row, max_col = self.bbox
            bbox = {
                "min_row": min_row,
                "min_col": min_col,
                "max_row": max_row,
                "max_col": max_col,
            }
        return {"has_mask": self.has_mask, "area": self.area, "bbox": bbox}


def mask_stats(mask: np.ndarray) -> MaskStats:
    """Compute the foreground area and inclusive bbox of a binary mask."""

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return MaskStats(has_mask=False, area=0, bbox=None)
    return MaskStats(
        has_mask=True,
        area=int(ys.size),
        bbox=(int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
    )


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a saved binary mask as a uint8 (0/255) numpy array."""

    image = Image.open(mask_path)
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8)


def save_mask(mask: np.ndarray, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)


def save_prompt_overlay(
    frame_rgb: np.ndarray,
    masks: Sequence[np.ndarray],
    labels: Sequence[str],
    output_path: Path,
    alpha: float,
    colors_rgb: Sequence[tuple[int, int, int]] = PROMPT_COLORS_RGB,
    overwrite: bool = True,
) -> None:
    """Save a multi-prompt mask overlay with a color-keyed legend."""

    if output_path.exists() and not overwrite:
        return
    if not 0 <= alpha <= 1:
        raise ValueError("--overlay-alpha must be within [0, 1].")
    if len(masks) != len(labels):
        raise ValueError("masks and labels must have the same length.")
    if len(masks) > len(colors_rgb):
        raise ValueError(f"At most {len(colors_rgb)} mask prompts are supported.")

    expected_shape = frame_rgb.shape[:2]
    vis = frame_rgb.astype(np.float32).copy()
    for mask, color in zip(masks, colors_rgb):
        if mask.shape != expected_shape:
            raise ValueError(
                f"Overlay mask shape mismatch: expected {expected_shape}, got {mask.shape}"
            )
        mask_bool = mask > 0
        mask_color = np.asarray(color, dtype=np.float32)
        vis[mask_bool] = vis[mask_bool] * (1 - alpha) + mask_color * alpha

    image = Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8))
    if len(labels) >= 2:
        draw = ImageDraw.Draw(image, "RGBA")
        font_size = max(14, round(min(frame_rgb.shape[:2]) * 0.026))
        font = ImageFont.load_default(size=font_size)
        padding = max(10, round(font_size * 0.45))
        swatch = font_size
        row_height = round(font_size * 1.45)
        text_widths = [draw.textbbox((0, 0), label, font=font)[2] for label in labels]
        panel_width = padding * 3 + swatch + max(text_widths, default=0)
        panel_height = padding * 2 + row_height * len(labels)
        margin = max(8, round(font_size * 0.55))
        draw.rounded_rectangle(
            (margin, margin, margin + panel_width, margin + panel_height),
            radius=max(5, round(font_size * 0.3)),
            fill=(0, 0, 0, 205),
        )
        for index, (label, color) in enumerate(zip(labels, colors_rgb)):
            y = margin + padding + index * row_height
            draw.rectangle(
                (
                    margin + padding,
                    y + 2,
                    margin + padding + swatch,
                    y + 2 + swatch,
                ),
                fill=(*color, 255),
                outline=(255, 255, 255, 230),
                width=max(1, round(font_size * 0.08)),
            )
            draw.text(
                (margin + padding * 2 + swatch, y),
                label,
                fill=(255, 255, 255, 255),
                font=font,
                stroke_width=1,
                stroke_fill=(0, 0, 0, 255),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def save_overlay(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    alpha: float,
    mask_color_rgb: tuple[int, int, int],
    overwrite: bool,
) -> None:
    """Save a legacy single-mask overlay without a legend."""

    save_prompt_overlay(
        frame_rgb,
        [mask],
        ["mask"],
        output_path,
        alpha,
        colors_rgb=[mask_color_rgb],
        overwrite=overwrite,
    )
