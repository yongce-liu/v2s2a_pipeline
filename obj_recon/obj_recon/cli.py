"""Command-line entry point for the obj_recon package.

Consumes the ``segment`` stage's ``masks.json`` directly (the same way
``segment`` consumes the ``process`` stage's ``frames.json``):

.. code-block:: bash

    # Reconstruct the object masks from the segment stage
    uv run python -m obj_recon.cli \
        --masks-json outputs/yellow_spoon/segment/masks.json \
        --geometry-json outputs/yellow_spoon/geometry/geometry.json \
        --prompt-id "yellow spoon"

    # Reconstruct all text-prompt objects (default), only first 10 frames
    uv run python -m obj_recon.cli \
        --masks-json outputs/yellow_spoon/segment/masks.json \
        --geometry-json outputs/yellow_spoon/geometry/geometry.json \
        --max-frames 10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from loguru import logger

from obj_recon.multiview import MultiViewArgs, reconstruct_multiview
from obj_recon.reconstruct import MeshReconArgs, reconstruct_video

# Registration/init frame shared with pose_estimation's ``--init-frame``.
DEFAULT_INIT_FRAME = 0


@dataclass
class ObjReconCliArgs:
    """Reconstruct meshes from segment masks and geometry point maps."""

    masks_json: Path
    """Path to the ``segment`` stage's ``masks.json`` (the mask manifest)."""

    geometry_json: Path
    """Path to the ``geometry`` stage's ``geometry.json``. Its precomputed
    point maps are passed to SAM3D, so obj_recon does not run MoGe itself."""

    frames_json: Path | None = None
    """Optional path to the ``process`` stage's ``frames.json``. If omitted,
    frame paths are resolved via the ``source_frames_json`` field in
    ``masks.json`` (falling back to ``../process/frames/``)."""

    output_root: Path | None = None
    """Root under which ``<clip>/obj_recon/`` is created. Defaults to the
    segment clip root (so outputs land next to ``segment/``)."""

    prompt_id: list[str] | None = None
    """Which ``prompt_masks[].prompt_id`` entries to reconstruct. Repeat the
    flag to select several. Defaults to all prompts whose ``input_type`` is
    ``"text"`` (i.e. the scene objects, not the hands)."""

    max_frames: int | None = None
    """Limit the number of frames processed (useful for smoke tests)."""

    frame_index: list[int] | None = None
    """Process only these frame indices. Repeat to select multiple frames.

    Defaults to the single registration/init frame (``[0]``). Because the
    rigid object's size never changes, one frame's mesh is sufficient —
    pose_estimation then tracks that fixed mesh every frame (the mesh supplies
    metric scale; per-frame geometry depth drives the tracker). Pass explicit
    indices (or use ``--max-frames``) only to reconstruct more frames, e.g. for
    debugging or when single-frame quality is poor."""

    skip_existing: bool = False
    """Skip frames whose ``layout.json`` already exists in the output dir."""

    recon: MeshReconArgs = field(default_factory=MeshReconArgs)
    """SAM 3D Objects reconstruction settings."""

    mv: MultiViewArgs = field(default_factory=MultiViewArgs)
    """Multi-view (MV-SAM3D) reconstruction settings. Enable with
    ``--mv.enabled`` (defaults to off → single-frame mode)."""


def main() -> None:
    args = tyro.cli(ObjReconCliArgs)

    if args.mv.enabled:
        # Multi-view (MV-SAM3D): fuse keyframes into one mesh per object and a
        # reference (view-0) metric pose; pose_estimation fills the rest.
        out = reconstruct_multiview(
            args.masks_json,
            geometry_json=args.geometry_json,
            frames_json=args.frames_json,
            output_root=args.output_root,
            prompt_ids=args.prompt_id,
            recon_args=args.recon,
            mv_args=args.mv,
            manual_frame_index=args.frame_index,
        )
        logger.info(
            "[obj_recon] multi-view complete: {} objects under {} (keyframes {})",
            len(out["objects"]),
            out["meshes_dir"],
            out["keyframes"],
        )
        return

    # Single-frame by default: the object is rigid, so one frame's mesh fixes
    # its (constant) size; the other frames only need poses, not new meshes.
    frame_indices = args.frame_index if args.frame_index else [DEFAULT_INIT_FRAME]

    outputs = reconstruct_video(
        args.masks_json,
        geometry_json=args.geometry_json,
        frames_json=args.frames_json,
        output_root=args.output_root,
        prompt_ids=args.prompt_id,
        max_frames=args.max_frames,
        frame_indices=frame_indices,
        skip_existing=args.skip_existing,
        args=args.recon,
    )
    logger.info(
        "[obj_recon] complete: {} frames processed, manifest at {}",
        len(outputs.entries),
        outputs.meshes_json_path,
    )


if __name__ == "__main__":
    main()
