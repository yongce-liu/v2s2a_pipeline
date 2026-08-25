"""Math utilities."""

import torch


def quat_xyzw2wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], dim=-1)


def quat_wxyz2xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat_wxyz[..., 0:3], quat_wxyz[..., 3:4]], dim=-1)


def quat_to_vel(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to angular velocity."""
    axis = quat[..., 1:4]
    sin_a_2 = torch.norm(axis, dim=-1, keepdim=True)

    zero_mask = sin_a_2[..., 0] == 0.0
    result = torch.zeros_like(axis)

    non_zero_mask = ~zero_mask
    if torch.any(non_zero_mask):
        speed = 2.0 * torch.atan2(sin_a_2[non_zero_mask, 0], quat[non_zero_mask, 0])
        # when axis-angle is larger than pi, rotation is in the opposite direction
        speed = torch.where(speed > torch.pi, speed - 2.0 * torch.pi, speed)

        result[non_zero_mask] = (
            axis[non_zero_mask] * speed[..., None] / sin_a_2[non_zero_mask]
        )

    return result


def mul_quat(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(u)
    result[..., 0] = (
        u[..., 0] * v[..., 0]
        - u[..., 1] * v[..., 1]
        - u[..., 2] * v[..., 2]
        - u[..., 3] * v[..., 3]
    )
    result[..., 1] = (
        u[..., 0] * v[..., 1]
        + u[..., 1] * v[..., 0]
        + u[..., 2] * v[..., 3]
        - u[..., 3] * v[..., 2]
    )
    result[..., 2] = (
        u[..., 0] * v[..., 2]
        - u[..., 1] * v[..., 3]
        + u[..., 2] * v[..., 0]
        + u[..., 3] * v[..., 1]
    )
    result[..., 3] = (
        u[..., 0] * v[..., 3]
        + u[..., 1] * v[..., 2]
        - u[..., 2] * v[..., 1]
        + u[..., 3] * v[..., 0]
    )
    return result


def quat_sub(qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
    """Angular difference between (w, x, y, z) quaternions qa and qb."""
    qneg = qb.clone()
    qneg[..., 1:] = -qneg[..., 1:]
    qdif = mul_quat(qneg, qa)
    return quat_to_vel(qdif)
