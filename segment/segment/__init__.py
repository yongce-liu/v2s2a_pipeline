"""Shared helpers for the segment package."""

__version__ = "0.1.0"

from segment.device import resolve_torch_device, set_cuda_device_if_indexed
from segment.media import (
    MaskStats,
    load_mask,
    load_rgb_image,
    mask_is_empty,
    mask_stats,
    save_mask,
    save_overlay,
    save_prompt_overlay,
)

__all__ = [
    "MaskStats",
    "load_mask",
    "load_rgb_image",
    "mask_is_empty",
    "mask_stats",
    "resolve_torch_device",
    "save_mask",
    "save_overlay",
    "save_prompt_overlay",
    "set_cuda_device_if_indexed",
]
