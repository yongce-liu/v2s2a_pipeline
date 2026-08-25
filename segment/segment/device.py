"""Device helpers for the segment package."""

from __future__ import annotations

import torch


def resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is not available: {device}")
    return torch_device


def set_cuda_device_if_indexed(device: torch.device) -> None:
    """Set current CUDA device only when the user selected a concrete index."""

    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
