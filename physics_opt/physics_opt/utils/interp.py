"""Define functions to interpolate the control signal."""

from __future__ import annotations

import loguru
import torch
import torch.nn.functional as F


def interp(src: torch.Tensor, n: int, order: int = 1) -> torch.Tensor:
    """Upsample (N, H, D) by factor n with order 0/1/2 hold -> (N, H*n, D)."""
    if order not in [0, 1, 2]:
        raise ValueError("Order must be an integer: 0, 1, or 2.")

    N, H, D = src.shape

    if H <= 1:
        return src.repeat(1, n, 1)

    if order == 0:
        mode = "nearest"
    elif order == 1:
        mode = "linear"
    elif order == 2:
        if H < 3:
            loguru.logger.warning(
                f"Source tensor has H={H} < 3 time steps. "
                "Falling back to linear interpolation for order=2."
            )
            mode = "linear"
        else:
            mode = "quadratic"

    if not src.is_floating_point():
        src = src.to(torch.float32)

    # F.interpolate wants (N, channels, length); treat D as channels, H as length.
    src_permuted = src.permute(0, 2, 1)

    dst_len = H * n

    # align_corners=True keeps endpoint values exact; n/a for 'nearest'.
    align = mode != "nearest"

    dst_permuted = F.interpolate(
        src_permuted, size=dst_len, mode=mode, align_corners=align
    )

    dst = dst_permuted.permute(0, 2, 1)

    return dst


def align_to_sim_dt(
    arr: torch.Tensor,
    ref_dt: float,
    sim_dt: float,
    order: int = 1,
) -> torch.Tensor:
    """Resample a (T, D) reference signal onto a sim_dt timeline."""
    if ref_dt >= sim_dt:
        ref_steps = max(1, int(round(ref_dt / sim_dt)))
        if ref_steps == 1:
            return arr
        return interp(arr.unsqueeze(0), ref_steps, order=order).squeeze(0)
    factor = max(1, int(round(sim_dt / ref_dt)))
    return arr[::factor]


def get_slice(
    src: tuple[torch.Tensor, ...], start: int, end: int
) -> tuple[torch.Tensor, ...]:
    return tuple(s[start:end] for s in src)
