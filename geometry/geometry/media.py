"""Image, depth, and point-cloud I/O helpers."""

from __future__ import annotations

import os

# MoGe's own scripts enable OpenEXR support the same way before importing cv2.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_mask(mask: np.ndarray, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)


def save_depth_exr(depth: np.ndarray, output_path: Path, overwrite: bool) -> None:
    """Save a float32 depth map as a single-channel EXR."""

    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_path),
        depth.astype(np.float32),
        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT],
    )


def save_points_ply(
    points: np.ndarray,
    colors_rgb: np.ndarray | None,
    output_path: Path,
    overwrite: bool,
) -> None:
    """Save a camera-space point cloud (H, W, 3) with optional vertex colors."""

    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = points.reshape(-1, 3).astype(np.float64)
    finite = np.isfinite(vertices).all(axis=1)
    colors = None
    if colors_rgb is not None:
        flat_colors = colors_rgb.reshape(-1, colors_rgb.shape[-1]).astype(np.uint8)[
            finite
        ]
        # trimesh expects float RGB in [0, 1] when passed as vertex_colors.
        colors = flat_colors / 255.0
    cloud = trimesh.points.PointCloud(vertices[finite], colors=colors)
    cloud.export(output_path)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Colorize a depth map for visualization (valid pixels only)."""

    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    values = depth[valid]
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    span = max(hi - lo, 1e-6)
    normalized = np.clip((values - lo) / span, 0.0, 1.0)
    # Simple inferno-like ramp: dark blue -> magenta -> yellow.
    colored = np.stack(
        [
            np.clip(1.5 * normalized - 0.5, 0.0, 1.0),
            np.clip(1.5 * normalized - 0.25, 0.0, 1.0),
            np.clip(0.5 * normalized + 0.5 * (1 - normalized), 0.0, 1.0),
        ],
        axis=-1,
    )
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    out[valid] = (colored * 255).astype(np.uint8)
    return out
