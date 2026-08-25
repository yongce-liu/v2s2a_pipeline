"""Thin wrapper around the upstream ``moge`` package.

Everything specific to this pipeline lives here: loading a local checkpoint,
per-frame inference, and converting the
model's tensors into numpy arrays with denormalized intrinsics (the format
downstream stages such as HaWoR expect, matching do-as-i-do's
``get_pointmap_dir.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger

from geometry import resolve_torch_device, set_cuda_device_if_indexed

DEFAULT_CHECKPOINT = Path(__file__).parents[2] / "weights/moge-3/moge-3-vitg/model.pt"
DEFAULT_MODEL_VERSION = "v3"
"""Default to the largest MoGe-3 checkpoint (ViT-giant, metric scale + refiner)."""


@dataclass
class MogeArgs:
    """Arguments for MoGe monocular geometry estimation."""

    checkpoint: Path = DEFAULT_CHECKPOINT
    """Local MoGe checkpoint directory or ``.pt`` file saved by ``save_pretrained``."""
    version: str = DEFAULT_MODEL_VERSION
    """Model family of the checkpoint: ``v1``, ``v2``, or ``v3``."""
    allow_hf_download: bool = False
    """Allow ``from_pretrained`` to fetch weights from Hugging Face."""
    device: str = "auto"
    fov_x: float | None = None
    """Horizontal field of view in degrees when known; otherwise estimated."""
    resolution_level: int = 9
    """Token budget level 0-9; higher is finer but slower."""
    num_tokens: int | None = None
    """Explicit token count; overrides ``resolution_level`` when set."""
    refine_steps: int = 3
    """Sparse-refinement steps for v3 only (ignored by other versions)."""
    use_fp16: bool = True
    force_projection: bool = False
    apply_mask: bool = True
    overwrite: bool = True


@dataclass(frozen=True)
class MogeFrameResult:
    """One image's geometry as numpy arrays."""

    points: np.ndarray
    """Camera-space point map ``(H, W, 3)``; invalid pixels are ``inf``."""
    depth: np.ndarray
    """Depth map ``(H, W)``; invalid pixels are ``inf``."""
    mask: np.ndarray | None
    """Valid-pixel boolean mask ``(H, W)``, or None if the model has no mask head."""
    normal: np.ndarray | None
    intrinsics: np.ndarray
    """Normalized 3x3 intrinsics (fx etc. in units of image width/height)."""

    def denormalized_intrinsics(self, width: int, height: int) -> np.ndarray:
        """Scale the normalized intrinsics to pixel units at this resolution."""

        scaled = self.intrinsics.copy()
        scaled[0, 0] *= width
        scaled[0, 2] *= width
        scaled[1, 1] *= height
        scaled[1, 2] *= height
        return scaled


class MogeModel:
    """Reusable MoGe geometry estimator built on the vendored package."""

    def __init__(self, args: MogeArgs) -> None:
        checkpoint = args.checkpoint.expanduser()
        if not checkpoint.exists():
            if not args.allow_hf_download:
                raise FileNotFoundError(
                    f"MoGe checkpoint not found: {checkpoint} "
                    "(pass --moge.allow-hf-download to download from Hugging Face)"
                )
            checkpoint = str(checkpoint)

        self.device = resolve_torch_device(args.device)
        set_cuda_device_if_indexed(self.device)
        self.args = args

        from moge.model import import_model_class_by_version

        logger.info(
            "[MoGe] Loading model: version={}, device={}, source={}",
            args.version,
            self.device,
            checkpoint,
        )
        model = (
            import_model_class_by_version(args.version)
            .from_pretrained(str(checkpoint))
            .to(self.device)
            .eval()
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def infer_image(
        self,
        frame_rgb: np.ndarray,
        fov_x: float | None = None,
    ) -> MogeFrameResult:
        """Estimate geometry for one RGB image (uint8 HWC numpy array)."""

        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"Expected an RGB image (H, W, 3), got {frame_rgb.shape}")

        args = self.args
        image_tensor = (
            torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        infer_kwargs: dict = {
            "resolution_level": args.resolution_level,
            "force_projection": args.force_projection,
            "apply_mask": args.apply_mask,
            "use_fp16": args.use_fp16,
        }
        effective_fov = fov_x if fov_x is not None else args.fov_x
        if effective_fov is not None:
            infer_kwargs["fov_x"] = effective_fov
        if args.num_tokens is not None:
            infer_kwargs["num_tokens"] = args.num_tokens
        if args.version == "v3":
            infer_kwargs["refine_steps"] = args.refine_steps

        output = self.model.infer(image_tensor.to(self.device), **infer_kwargs)
        return MogeFrameResult(
            points=output["points"].float().cpu().numpy(),
            depth=output["depth"].float().cpu().numpy(),
            mask=(
                output["mask"].detach().cpu().numpy().astype(bool)
                if "mask" in output
                else None
            ),
            normal=(
                output["normal"].float().cpu().numpy() if "normal" in output else None
            ),
            intrinsics=output["intrinsics"].float().cpu().numpy(),
        )

    def infer_batch(
        self,
        frames_rgb: list[np.ndarray],
        fov_x: float | None = None,
    ) -> list[MogeFrameResult]:
        """Infer several frames one at a time with the shared loaded model."""
        return [self.infer_image(frame, fov_x=fov_x) for frame in frames_rgb]


def load_moge_model(args: MogeArgs) -> MogeModel:
    """Public loader so callers can construct the model once and reuse it."""

    return MogeModel(args)


if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    cli_args = tyro.cli(MogeArgs)
    load_moge_model(cli_args)
