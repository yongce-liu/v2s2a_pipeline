"""Image and mask I/O helpers for the obj_recon package."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array (H, W, 3)."""
    image_path = Path(image_path).expanduser()
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a binary mask as a boolean numpy array (H, W).

    Reads a PNG mask (grayscale or RGBA) and thresholds at > 0.
    """
    mask_path = Path(mask_path).expanduser()
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    image = Image.open(mask_path)
    arr = np.asarray(image)
    if arr.ndim == 3:
        # Use alpha channel if available, else luminance
        if arr.shape[2] == 4:
            arr = arr[..., 3]
        else:
            arr = np.mean(arr[..., :3], axis=2)
    return (arr > 0).astype(bool)


def load_rgba_image(image_path: Path) -> np.ndarray:
    """Load an image as RGBA uint8 numpy array (H, W, 4).

    If the source has no alpha channel, an opaque one (255) is appended.
    """
    image_path = Path(image_path).expanduser()
    rgb = load_rgb_image(image_path)
    # Try loading with PIL to preserve alpha if present
    pil_img = Image.open(image_path)
    if pil_img.mode == "RGBA":
        return np.asarray(pil_img, dtype=np.uint8)
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    return np.dstack([rgb, alpha])


def mask_to_alpha(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean mask to a uint8 alpha channel (0 or 255)."""
    return mask.astype(np.uint8) * 255


def mask_is_empty(mask: np.ndarray) -> bool:
    """Return True if the mask has no foreground pixels."""
    return not np.any(mask > 0)
