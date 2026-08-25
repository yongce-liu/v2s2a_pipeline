"""SAM3D pipeline variant that requires externally computed geometry."""

from __future__ import annotations

import torch
from sam3d_objects.pipeline.inference_pipeline_pointmap import (
    InferencePipelinePointMap,
)


class PrecomputedPointMapPipeline(InferencePipelinePointMap):
    """Use point maps and intrinsics supplied by the ``geometry`` stage.

    The upstream point-map pipeline accepts an external point map but discards
    its known intrinsics and re-infers them. This local subclass keeps the
    upstream model and preprocessing behavior while replacing only that input
    adapter. The inherited ``depth_model`` is always ``None`` and is never used.
    """

    def compute_pointmap(self, image, pointmap=None):
        if not isinstance(pointmap, dict):
            raise TypeError(
                "Expected precomputed geometry with 'points' and 'intrinsics'; "
                "run the geometry stage first."
            )

        points = pointmap.get("points")
        intrinsics = pointmap.get("intrinsics")
        if not isinstance(points, torch.Tensor) or not isinstance(
            intrinsics, torch.Tensor
        ):
            raise TypeError("Precomputed points and intrinsics must be torch tensors.")

        loaded_image = torch.from_numpy(self.image_to_float(image))
        loaded_mask = loaded_image[..., -1]
        loaded_image = loaded_image.permute(2, 0, 1).contiguous()[:3]

        if points.shape != loaded_image.permute(1, 2, 0).shape:
            raise ValueError(
                f"Point map shape {tuple(points.shape)} does not match image "
                f"shape {tuple(loaded_image.shape[1:])}."
            )
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Expected normalized 3x3 intrinsics, got {tuple(intrinsics.shape)}."
            )

        points = points.to(self.device)
        intrinsics = intrinsics.to(self.device)
        points = self._clip_pointmap(points.permute(2, 0, 1), loaded_mask)
        return {
            "pts_color": loaded_image,
            "intrinsics": intrinsics,
            "pointmap": points,
        }
