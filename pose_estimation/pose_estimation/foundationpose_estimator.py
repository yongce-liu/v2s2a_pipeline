"""Thin wrapper around the vendored ``pkgs/FoundationPose`` estimator.

``pkgs/FoundationPose`` is never modified (the only additions there are an
install ``pyproject.toml`` and empty ``learning/**/__init__.py`` files).
Everything specific to this pipeline lives here:

- lazy imports: ``estimater`` pulls in the compiled ``mycpp``/``nvdiffrast``
  extensions and the GPU weights, so it is imported only when a model is
  actually constructed;
- configurable weight resolution, defaulting to the pipeline's
  ``weights/foundationpose`` directory;
- a pipeline-friendly ``register()`` / ``track()`` API on top of the upstream
  ``register`` / ``track_one`` methods, following the calling pattern proven
  out in do-as-i-do's ``track_object_foundationpose.py`` and the upstream
  ``run_demo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from loguru import logger

from pose_estimation import resolve_torch_device, set_cuda_device_if_indexed

FOUNDATIONPOSE_ROOT = Path(__file__).parents[2] / "pkgs" / "FoundationPose"
DEFAULT_WEIGHTS_ROOT = Path(__file__).parents[2] / "weights" / "foundationpose"

SCORER_RUN_NAME = "2024-01-11-20-02-45"
REFINER_RUN_NAME = "2023-10-28-18-33-37"


@dataclass
class FoundationPoseArgs:
    """Arguments for FoundationPose model construction and inference."""

    weights_root: Path = DEFAULT_WEIGHTS_ROOT
    """Directory holding ``<run_name>/{model_best.pth, config.yml}``."""

    device: str = "cuda"
    """Torch device; FoundationPose's networks are CUDA-only."""

    est_refine_iter: int = 5
    """Refinement iterations for the initial pose registration."""

    track_refine_iter: int = 10
    """Refinement iterations for per-frame tracking."""

    track_crop_ratio: float | None = 2.0
    """Override the refiner crop ratio used during tracking. Larger values tolerate
    faster inter-frame motion at the cost of a less focused crop."""

    debug: int = 0
    """Upstream debug level (0 = silent; >=2 writes render debug artifacts)."""

    debug_dir: Path | None = None
    """Where upstream debug artifacts go; defaults to a temp dir under the stage."""

    overwrite: bool = True

    extra_env: dict[str, str] = field(default_factory=dict)
    """Reserved for future env overrides (unused for now)."""


def ensure_importable(weights_root: Path = DEFAULT_WEIGHTS_ROOT) -> None:
    """Verify the vendored FoundationPose package and its weights are usable.

    Raises with a actionable message before any heavyweight import happens.
    """

    if not (FOUNDATIONPOSE_ROOT / "estimater.py").exists():
        raise FileNotFoundError(
            f"FoundationPose checkout not found at {FOUNDATIONPOSE_ROOT}"
        )

    weights_root = weights_root.expanduser().resolve()
    if not weights_root.exists():
        raise FileNotFoundError(
            f"FoundationPose weights not found at {weights_root}. "
            f"Expected '<weights_root>/{SCORER_RUN_NAME}' and "
            f"'<weights_root>/{REFINER_RUN_NAME}'."
        )
    for run_name in (SCORER_RUN_NAME, REFINER_RUN_NAME):
        for filename in ("model_best.pth", "config.yml"):
            candidate = weights_root / run_name / filename
            if not candidate.exists():
                raise FileNotFoundError(
                    f"FoundationPose weight file missing: {candidate}"
                )

    try:
        # Utils.py inserts mycpp/build into sys.path on import, so import it
        # first, then check that the extension actually resolved.
        import Utils  # noqa: F401, I001
        import mycpp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FoundationPose's mycpp extension is not built. "
            "Run: cd pkgs/FoundationPose/mycpp && mkdir -p build && cd build "
            "&& cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j"
        ) from exc

    try:
        import nvdiffrast  # noqa: F401
        import pytorch3d  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FoundationPose needs the nvdiffrast and pytorch3d GPU extensions, "
            "both built against the installed torch. Install them with "
            "pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git "
            "and git+https://github.com/facebookresearch/pytorch3d.git"
        ) from exc


class FoundationPoseEstimator:
    """Reusable FoundationPose pose estimator for one object mesh.

    Wraps the upstream ``FoundationPose`` class. The mesh is centered the same
    way ``run_demo.py`` does it (via ``trimesh.bounds.oriented_bounds`` for the
    visualization bbox), and the scorer/refiner networks are loaded once and
    shared across all frames.
    """

    def __init__(self, mesh: trimesh.Trimesh, args: FoundationPoseArgs) -> None:
        weights_root = args.weights_root.expanduser().resolve()
        ensure_importable(weights_root)

        self.device = resolve_torch_device(args.device)
        if self.device.type != "cuda":
            raise RuntimeError(
                "FoundationPose requires a CUDA device (its networks and "
                "nvdiffrast context are CUDA-only)."
            )
        set_cuda_device_if_indexed(self.device)
        self.args = args

        # Lazy, heavy imports: estimater pulls in torch models + mycpp.
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
        from learning.training.predict_pose_refine import PoseRefinePredictor as _PR
        from learning.training.predict_score import ScorePredictor as _SP

        self._fp_classes = (FoundationPose, ScorePredictor, PoseRefinePredictor)
        self._predictor_classes = (_SP, _PR)

        logger.info(
            "[FoundationPose] Loading scorer ({}) and refiner ({}) weights",
            SCORER_RUN_NAME,
            REFINER_RUN_NAME,
        )
        scorer = ScorePredictor(weights_dir=str(weights_root))
        refiner = PoseRefinePredictor(weights_dir=str(weights_root))
        if args.track_crop_ratio is not None:
            if args.track_crop_ratio <= 0:
                raise ValueError("track_crop_ratio must be positive")
            refiner.cfg["crop_ratio"] = float(args.track_crop_ratio)
            logger.info(
                "[FoundationPose] tracking crop ratio overridden to {:.2f}",
                args.track_crop_ratio,
            )

        import nvdiffrast.torch as dr

        glctx = dr.RasterizeCudaContext()

        debug_dir = args.debug_dir
        if debug_dir is None:
            import tempfile

            debug_dir = Path(tempfile.mkdtemp(prefix="foundationpose_debug_"))
        debug_dir.mkdir(parents=True, exist_ok=True)

        self.mesh = mesh
        self.estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=args.debug,
            debug_dir=str(debug_dir),
        )

        # Bbox of the *centered* mesh in FoundationPose's own convention
        # (reset_object re-centers mesh.vertices around self.model_center).
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        self.to_origin = to_origin

        self._registered = False

    # ------------------------------------------------------------------ API

    def register(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """Estimate the initial 4x4 object pose (ob_in_cam) from one frame."""

        try:
            pose = self.estimator.register(
                K=intrinsics,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=self.args.est_refine_iter,
            )
            result = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        finally:
            self._release_registration_temporaries()
        self._registered = True
        return result

    def _release_registration_temporaries(self) -> None:
        """Release upstream registration hypotheses retained for debugging."""

        pose_last = getattr(self.estimator, "pose_last", None)
        if pose_last is not None and hasattr(pose_last, "detach"):
            self.estimator.pose_last = pose_last.detach().clone()

        best_id = getattr(self.estimator, "best_id", None)
        if best_id is not None and hasattr(best_id, "detach"):
            self.estimator.best_id = best_id.detach().clone()

        self.estimator.poses = None
        self.estimator.scores = None

    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """Track the object into the next frame (requires register() first)."""

        if not self._registered:
            raise RuntimeError("Call register() on the reference frame before track().")
        pose = self.estimator.track_one(
            rgb=rgb,
            depth=depth,
            K=intrinsics,
            iteration=self.args.track_refine_iter,
        )
        return np.asarray(pose, dtype=np.float64).reshape(4, 4)

    def shift_pose_translation(self, delta_xyz: np.ndarray) -> None:
        """Shift the current original-mesh pose translation before refinement."""

        if not self._registered:
            raise RuntimeError("Cannot shift pose before registration.")
        import torch

        delta = torch.as_tensor(
            np.asarray(delta_xyz, dtype=np.float32).reshape(3),
            device=self.estimator.pose_last.device,
            dtype=self.estimator.pose_last.dtype,
        )
        pose_last = self.estimator.pose_last.detach().clone()
        if pose_last.ndim == 2:
            pose_last[:3, 3] += delta
        elif pose_last.ndim == 3:
            pose_last[:, :3, 3] += delta[None]
        else:
            raise ValueError(f"Unexpected pose_last shape: {tuple(pose_last.shape)}")
        self.estimator.pose_last = pose_last

    def set_pose(self, pose: np.ndarray) -> None:
        """Seed tracking with an externally computed pose (do-as-i-do pattern).

        The upstream refiner keeps its own ``pose_last`` in *centered-mesh*
        coordinates, so an externally supplied camera-frame pose must be
        shifted back by the mesh center.
        """

        import torch

        original = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        to_original_mesh = np.eye(4, dtype=np.float32)
        to_original_mesh[:3, 3] = self.estimator.model_center.reshape(3)
        centered = original @ to_original_mesh
        self.estimator.pose_last = torch.as_tensor(centered, device="cuda")
        self._registered = True

    @property
    def model_center(self) -> np.ndarray:
        """Center of the original (un-centered) mesh, from upstream."""

        return np.asarray(self.estimator.model_center, dtype=np.float64)
