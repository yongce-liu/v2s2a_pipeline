"""Gravity estimation using GeoCalib for the scene_construction package.

The do-as-i-do reconstruction pipeline writes this under
``raw_dir/stage2_gravity/gravity.json``; the v2s2a pipeline has no separate
reconstruction stage, so scene_construction estimates it directly from the
``process`` stage frames and caches the result beside them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(
        path for path in Path(frames_dir).iterdir() if path.suffix.lower() in IMAGE_EXTS
    )


@dataclass(frozen=True)
class GravityEstimate:
    vec3d: list[float]
    roll_deg: float
    pitch_deg: float
    n_frames: int
    n_inliers: int


def _angle_between(v1, v2):
    dot = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    return torch_acos(dot)


def torch_acos(x):
    import torch

    return torch.acos(x)


def estimate_gravity_from_frames(
    frames_dir: Path,
    camera_model: str = "pinhole",
    max_frames: int | None = 32,
    device: str = "cuda",
) -> GravityEstimate:
    """Estimate world-up in the camera frame from a directory of RGB frames."""
    try:
        import torch
        from geocalib import GeoCalib
        from geocalib.gravity import Gravity
    except ImportError as exc:
        raise RuntimeError(
            "GeoCalib + torch are required for gravity estimation. Install them "
            "in the scene_construction environment and retry."
        ) from exc

    weights = "pinhole" if camera_model == "pinhole" else "distorted"
    model = GeoCalib(weights=weights).to(device)
    model.eval()

    image_paths = _list_frame_paths(frames_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image files found under {frames_dir}")
    if max_frames is not None and len(image_paths) > max_frames:
        idx = np.round(np.linspace(0, len(image_paths) - 1, max_frames)).astype(int)
        image_paths = [image_paths[i] for i in idx]

    vecs: list = []
    confs: list[float] = []
    with torch.no_grad():
        for path in image_paths:
            img = model.load_image(str(path)).to(device)
            result = model.calibrate(img, camera_model=camera_model)
            grav = result["gravity"]
            conf = float(
                (
                    result["up_confidence"].mean().item()
                    + result["latitude_confidence"].mean().item()
                )
                / 2.0
            )
            vecs.append(grav.vec3d.squeeze(0).cpu())
            confs.append(conf)
            logger.info(
                "  [{}] roll={:.1f}° pitch={:.1f}° conf={:.3f}",
                path.name,
                np.rad2deg(grav.roll.item()),
                np.rad2deg(grav.pitch.item()),
                conf,
            )

    if not vecs:
        raise RuntimeError(f"No valid frames under {frames_dir} for gravity estimation")

    vecs_t = torch.stack(vecs)
    confs_t = torch.tensor(confs)

    def spherical_mean(v, w=None):
        if w is not None:
            w = w / w.sum()
            m = (v * w.unsqueeze(-1)).sum(dim=0)
        else:
            m = v.mean(dim=0)
        return torch.nn.functional.normalize(m, dim=-1)

    mean_vec = spherical_mean(vecs_t)
    angles = _angle_between(vecs_t, mean_vec.unsqueeze(0).expand_as(vecs_t))
    median_angle = angles.median()
    mad = (angles - median_angle).abs().median().clamp(min=1e-6)
    threshold = median_angle + 3.0 * mad
    inlier_mask = angles <= threshold
    n_inliers = int(inlier_mask.sum().item())
    logger.info(
        "[gravity] outlier rejection: {}/{} kept (threshold={:.2f}°, median={:.2f}°)",
        n_inliers,
        len(vecs_t),
        np.rad2deg(threshold.item()),
        np.rad2deg(median_angle.item()),
    )

    final_vec = spherical_mean(vecs_t[inlier_mask], w=confs_t[inlier_mask])
    final_gravity = Gravity(final_vec.unsqueeze(0).to(device))
    roll_deg = float(np.rad2deg(final_gravity.roll.item()))
    pitch_deg = float(np.rad2deg(final_gravity.pitch.item()))
    logger.info(
        "[gravity] final: roll={:+.2f}° pitch={:+.2f}° vec3d={}",
        roll_deg,
        pitch_deg,
        np.round(final_vec.numpy(), 4).tolist(),
    )

    return GravityEstimate(
        vec3d=np.asarray(final_vec.numpy(), dtype=np.float64).tolist(),
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        n_frames=len(vecs),
        n_inliers=n_inliers,
    )


def gravity_dict(estimate: GravityEstimate) -> dict:
    """JSON payload matching the do-as-i-do gravity cache schema."""

    return {
        "vec3d": [float(v) for v in estimate.vec3d],
        "roll_deg": estimate.roll_deg,
        "pitch_deg": estimate.pitch_deg,
        "n_frames": estimate.n_frames,
        "n_inliers": estimate.n_inliers,
    }


def write_gravity_json(path: Path, estimate: GravityEstimate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(gravity_dict(estimate), indent=2) + "\n", encoding="utf-8"
    )
