"""Shared helpers for the pose_estimation package."""

__version__ = "0.1.0"

from pose_estimation.device import resolve_torch_device, set_cuda_device_if_indexed
from pose_estimation.media import (
    PoseVideoWriter,
    load_intrinsics,
    load_mask,
    load_rgb_image,
    render_mesh_overlay,
)

__all__ = [
    "PoseVideoWriter",
    "load_intrinsics",
    "load_mask",
    "load_rgb_image",
    "render_mesh_overlay",
    "resolve_torch_device",
    "set_cuda_device_if_indexed",
]
