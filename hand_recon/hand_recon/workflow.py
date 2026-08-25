"""End-to-end HaWoR hand reconstruction for one clip.

Reads the ``process`` stage's frame manifest, runs HaWoR's detect-track,
motion estimation, optional SLAM, infiller, and MANO forward pass, and writes
the agreed pipeline artifacts under ``<clip_root>/hand_recon/``:

.. code-block:: text

    <clip_root>/hand_recon/
    ├── config.json
    ├── hands.json            # published hand-reconstruction metadata
    ├── hand_anchors.json     # per-frame HaWoR left/right 2D anchors
    ├── meshes.npz            # per-frame left/right vertices/joints/faces
    ├── vis/
    │   ├── aitviewer/
    │   └── overlay.mp4
    └── ...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from hand_recon import __version__
from hand_recon.anchors import HAND_ANCHORS_FILENAME, build_hand_anchors_payload
from hand_recon.hawor_stage import (
    HandReconHaworArgs,
    ensure_hawor_importable,
    hawor_workspace_env,
    run_detect_track,
    run_infiller,
    run_motion_estimation,
    run_slam,
)
from hand_recon.paths import (
    CAMERA_RIGHT_UP_TO_OPENCV,
    HAWOR_SOURCE,
)


@dataclass
class HandReconArgs:
    """Arguments for HaWoR-based hand reconstruction of one clip."""

    frames_json: Path
    """Path to the ``process`` stage ``frames.json`` frame manifest."""

    output_root: Path = Path(__file__).parents[2] / "outputs"
    """Root under which ``<clip_stem>/hand_recon/`` is created."""

    weights_root: Path | None = None
    """Root holding HaWoR's ``weights/{hawor,external}`` checkpoints; defaults
    to ``pkgs/HaWoR/weights`` but must contain the downloaded assets."""

    checkpoint: Path | None = None
    infiller_weight: Path | None = None

    img_focal: float | None = None
    """Image focal length in pixels; ``None`` falls back to HaWoR's estimate file or 600."""

    static_camera: bool = True
    """If True, skip DROID-SLAM and use identity world<->camera transforms."""

    vis: bool = True
    """Render the aitviewer overlay video for the reconstructed hands."""

    max_frames: int | None = None
    """Stop after N frames of the manifest (debug aid)."""


@dataclass(frozen=True)
class HandReconOutputs:
    clip_root: Path
    stage_dir: Path
    config_json_path: Path
    hands_json_path: Path
    hand_anchors_json_path: Path
    meshes_npz_path: Path
    vis_overlay_mp4: Path | None


_HAND2IDX = {"left": 0, "right": 1}


def _resolve_weights(args: HandReconArgs) -> tuple[Path, Path]:
    """Locate hawor.ckpt and infiller.pt, preferring explicit overrides."""
    if args.checkpoint is not None and args.infiller_weight is not None:
        return args.checkpoint, args.infiller_weight

    weights_root = args.weights_root or (HAWOR_SOURCE / "weights")
    checkpoint = args.checkpoint or (
        weights_root / "hawor" / "checkpoints" / "hawor.ckpt"
    )
    infiller = args.infiller_weight or (
        weights_root / "hawor" / "checkpoints" / "infiller.pt"
    )

    missing = [p for p in (checkpoint, infiller) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"HaWoR checkpoint(s) missing under {weights_root}: {missing}. "
            "Point --weights-root at a directory like do-as-i-do/weights/hawor/ "
            "or pass --checkpoint / --infiller-weight explicitly."
        )
    return checkpoint, infiller


def _hawor_args_dict(args: HandReconArgs) -> HandReconHaworArgs:
    checkpoint, infiller = _resolve_weights(args)
    return HandReconHaworArgs(
        video_path=args.frames_json,  # overwritten below once frames dir derived
        checkpoint=checkpoint,
        infiller_weight=infiller,
        img_focal=args.img_focal,
        static_camera=args.static_camera,
        vis_mode="cam",
    )


def _load_manifest(frames_json: Path) -> dict:
    return json.loads(frames_json.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_hand_recon(args: HandReconArgs) -> HandReconOutputs:
    """Run the full HaWoR pipeline for one clip and publish its outputs."""

    # HaWoR checkpoints ship omegaconf DictConfig pickles, and PyTorch ≥ 2.6
    # refuses them under weights_only=True by default. Restore the legacy
    # behavior for HaWoR's official weights; do-as-i-do does the same via
    # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 in the hawor conda env.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    ensure_hawor_importable()

    frames_json = args.frames_json.expanduser().resolve()
    manifest = _load_manifest(frames_json)

    clip_stem = frames_json.parent.parent.name
    clip_root = args.output_root.expanduser().resolve() / clip_stem
    stage_dir = clip_root / "hand_recon"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Stage a scratch video file from the manifest frames if no real video is
    # handy: HaWoR's entry point expects ``--video_path <file>`` but its
    # ``resolve_paths`` skips ffmpeg when the input is a directory of frames.
    frames_dir = frames_json.parent / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"process-stage frames missing: {frames_dir}")

    # Hand the frames dir to HaWoR through its ``video_path`` slot; the
    # detect_track stage treats dir-vs-file via resolve_paths.
    hawor_args = _hawor_args_dict(args)
    hawor_args.video_path = frames_dir

    with hawor_workspace_env(stage_dir) as workspace:
        start_idx, end_idx, seq_folder, imgfiles = run_detect_track(
            hawor_args, workspace
        )
        tracks_path = seq_folder / f"tracks_{start_idx}_{end_idx}" / "model_tracks.npy"
        anchors_payload = build_hand_anchors_payload(
            tracks_path,
            {
                **manifest,
                "source_frames_json": str(frames_json),
            },
            start_idx,
            end_idx,
        )
        hand_anchors_json_path = stage_dir / HAND_ANCHORS_FILENAME
        _write_json(hand_anchors_json_path, anchors_payload)

        frame_chunks_all, img_focal = run_motion_estimation(
            hawor_args, workspace, start_idx, end_idx, seq_folder
        )

        if args.static_camera:
            num = end_idx - start_idx
            zero_t = torch.zeros(num, 3)
            eye = torch.eye(3).repeat(num, 1, 1)
            R_w2c_sla_all, R_c2w_sla_all = eye, eye
            t_w2c_sla_all, t_c2w_sla_all = zero_t, zero_t
        else:
            R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = run_slam(
                hawor_args, workspace, start_idx, end_idx
            )

        (
            pred_trans,
            pred_rot,
            pred_hand_pose,
            pred_betas,
            pred_valid,
        ) = run_infiller(hawor_args, workspace, start_idx, end_idx, frame_chunks_all)

        return _publish_outputs(
            args=args,
            stage_dir=stage_dir,
            clip_root=clip_root,
            manifest=manifest,
            imgfiles=imgfiles,
            img_focal=img_focal,
            seq_folder=seq_folder,
            workspace=workspace,
            hand_anchors_json_path=hand_anchors_json_path,
            pred_trans=pred_trans,
            pred_rot=pred_rot,
            pred_hand_pose=pred_hand_pose,
            pred_betas=pred_betas,
            pred_valid=pred_valid,
            R_w2c_sla_all=R_w2c_sla_all,
            t_w2c_sla_all=t_w2c_sla_all,
            R_c2w_sla_all=R_c2w_sla_all,
            t_c2w_sla_all=t_c2w_sla_all,
        )


def _publish_outputs(
    args: HandReconArgs,
    stage_dir: Path,
    clip_root: Path,
    manifest: dict,
    imgfiles: list,
    img_focal: float,
    seq_folder: Path,
    workspace,
    hand_anchors_json_path: Path,
    pred_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    pred_hand_pose: torch.Tensor,
    pred_betas: torch.Tensor,
    pred_valid: torch.Tensor,
    R_w2c_sla_all: torch.Tensor,
    t_w2c_sla_all: torch.Tensor,
    R_c2w_sla_all: torch.Tensor,
    t_c2w_sla_all: torch.Tensor,
) -> HandReconOutputs:
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left

    vis_start = 0
    vis_end = pred_trans.shape[1]

    faces_right = np.concatenate([get_mano_faces(), _extra_wrist_faces()], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    idx_r = _HAND2IDX["right"]
    idx_l = _HAND2IDX["left"]

    pred_glob_r = run_mano(
        pred_trans[idx_r : idx_r + 1, vis_start:vis_end],
        pred_rot[idx_r : idx_r + 1, vis_start:vis_end],
        pred_hand_pose[idx_r : idx_r + 1, vis_start:vis_end],
        betas=pred_betas[idx_r : idx_r + 1, vis_start:vis_end],
    )
    right_verts = pred_glob_r["vertices"][0]
    right_joints = pred_glob_r["joints"][0]

    pred_glob_l = run_mano_left(
        pred_trans[idx_l : idx_l + 1, vis_start:vis_end],
        pred_rot[idx_l : idx_l + 1, vis_start:vis_end],
        pred_hand_pose[idx_l : idx_l + 1, vis_start:vis_end],
        betas=pred_betas[idx_l : idx_l + 1, vis_start:vis_end],
    )
    left_verts = pred_glob_l["vertices"][0]
    left_joints = pred_glob_l["joints"][0]

    R_x = torch.tensor(CAMERA_RIGHT_UP_TO_OPENCV).float()
    R_c2w_sla_all = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

    left_verts = torch.einsum("ij,tnj->tni", R_x, left_verts.cpu())
    right_verts = torch.einsum("ij,tnj->tni", R_x, right_verts.cpu())
    left_joints = torch.einsum("ij,tnj->tni", R_x, left_joints.cpu())
    right_joints = torch.einsum("ij,tnj->tni", R_x, right_joints.cpu())

    sl = slice(vis_start, vis_end)
    R_w2c, t_w2c = R_w2c_sla_all[sl], t_w2c_sla_all[sl]
    left_vertices_np = (
        (torch.einsum("bij,bvj->bvi", R_w2c, left_verts) + t_w2c[:, None]).cpu().numpy()
    )
    right_vertices_np = (
        (torch.einsum("bij,bvj->bvi", R_w2c, right_verts) + t_w2c[:, None])
        .cpu()
        .numpy()
    )
    left_joints_np = (
        (torch.einsum("bij,bnj->bni", R_w2c, left_joints) + t_w2c[:, None])
        .cpu()
        .numpy()
    )
    right_joints_np = (
        (torch.einsum("bij,bnj->bni", R_w2c, right_joints) + t_w2c[:, None])
        .cpu()
        .numpy()
    )

    meshes_path = stage_dir / "meshes.npz"
    np.savez(
        meshes_path,
        left_vertices=left_vertices_np,
        left_faces=faces_left.astype(np.int32),
        left_joints=left_joints_np,
        left_trans=pred_trans[idx_l, sl].cpu().numpy(),
        left_rot=pred_rot[idx_l, sl].cpu().numpy(),
        left_hand_pose=pred_hand_pose[idx_l, sl].cpu().numpy(),
        left_betas=pred_betas[idx_l, sl].cpu().numpy(),
        left_valid=np.asarray(pred_valid[idx_l, sl]),
        right_vertices=right_vertices_np,
        right_faces=faces_right.astype(np.int32),
        right_joints=right_joints_np,
        right_trans=pred_trans[idx_r, sl].cpu().numpy(),
        right_rot=pred_rot[idx_r, sl].cpu().numpy(),
        right_hand_pose=pred_hand_pose[idx_r, sl].cpu().numpy(),
        right_betas=pred_betas[idx_r, sl].cpu().numpy(),
        right_valid=np.asarray(pred_valid[idx_r, sl]),
    )

    vis_overlay_path: Path | None = None
    if args.vis:
        vis_overlay_path = _render_vis(
            stage_dir=stage_dir,
            imgfiles=imgfiles,
            img_focal=img_focal,
            right_verts=right_verts,
            right_faces=faces_right,
            left_verts=left_verts,
            left_faces=faces_left,
            R_w2c=R_w2c_sla_all[sl],
            t_w2c=t_w2c_sla_all[sl],
            vis_start=vis_start,
            vis_end=vis_end,
        )

    hands_json_path = stage_dir / "hands.json"
    _write_json(
        hands_json_path,
        _hands_json_dict(
            args,
            manifest,
            meshes_path,
            hand_anchors_json_path,
            img_focal,
        ),
    )

    config_json_path = stage_dir / "config.json"
    _write_json(config_json_path, _config_dict(args, manifest))

    logger.info(
        "[hand_recon] Done: frames={} meshes={} overlay={}",
        len(imgfiles),
        meshes_path,
        vis_overlay_path,
    )

    return HandReconOutputs(
        clip_root=clip_root,
        stage_dir=stage_dir,
        config_json_path=config_json_path,
        hands_json_path=hands_json_path,
        hand_anchors_json_path=hand_anchors_json_path,
        meshes_npz_path=meshes_path,
        vis_overlay_mp4=vis_overlay_path,
    )


def _extra_wrist_faces() -> np.ndarray:
    """Additional wrist-closure faces added to MANO for rendering (daid version)."""

    return np.array(
        [
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79],
        ]
    )


def _render_vis(
    stage_dir: Path,
    imgfiles: list,
    img_focal: float,
    right_verts: torch.Tensor,
    right_faces: np.ndarray,
    left_verts: torch.Tensor,
    left_faces: np.ndarray,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    vis_start: int,
    vis_end: int,
) -> Path | None:
    """Run HaWoR's aitviewer overlay and convert its frames to an mp4."""

    right_dict = {"vertices": right_verts.unsqueeze(0), "faces": right_faces}
    left_dict = {"vertices": left_verts.unsqueeze(0), "faces": left_faces}

    output_pth = stage_dir / f"vis_{vis_start}_{vis_end}"
    output_pth.mkdir(parents=True, exist_ok=True)
    image_names = imgfiles[vis_start:vis_end]

    from hand_recon.vis import run_video_cam_headless

    run_video_cam_headless(
        left_dict,
        right_dict,
        output_pth,
        img_focal,
        image_names,
        R_w2c=R_w2c,
        t_w2c=t_w2c,
    )

    aitviewer_dir = output_pth / "aitviewer"
    # Headless render_types=["video"] produces a single ``video.mp4``
    # alongside the per-frame images; reuse it directly as the overlay.
    produced = aitviewer_dir / "video.mp4"
    if not produced.exists():
        produced = next(
            (
                p
                for p in aitviewer_dir.iterdir()
                if p.name.startswith("video") and p.suffix == ".mp4"
            ),
            None,
        )
    if produced is None:
        logger.warning("[hand_recon] aitviewer output missing; skipping mp4")
        return None

    target = output_pth / "overlay.mp4"
    produced.replace(target)
    return target


def _hands_json_dict(
    args: HandReconArgs,
    manifest: dict,
    meshes_path: Path,
    hand_anchors_json_path: Path,
    img_focal: float,
) -> dict:
    return {
        "schema_version": "1.0",
        "stage": "hand_recon",
        "source_frames_json": str(args.frames_json.expanduser().resolve()),
        "source_video": manifest.get("source_video"),
        "fps": manifest.get("fps"),
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "img_focal": img_focal,
        "img_center": [manifest.get("width", 0) / 2, manifest.get("height", 0) / 2],
        "static_camera": args.static_camera,
        "meshes_npz": str(meshes_path),
        "hand_anchors_json": str(hand_anchors_json_path),
        "notation": {
            "rows": "rows are MANO joints 0..20 and mesh vertices; "
            "camera-space outputs use CV (x-right, y-down, z-forward); "
            "world outputs already include the right->up flip"
        },
    }


def _config_dict(args: HandReconArgs, manifest: dict) -> dict:
    return {
        "package": {"name": "hand_recon", "version": __version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "frames_json": str(args.frames_json.expanduser().resolve()),
            "source_video": manifest.get("source_video"),
            "fps": manifest.get("fps"),
        },
        "hand_recon": {
            "weights_root": (
                str(args.weights_root.expanduser()) if args.weights_root else None
            ),
            "checkpoint": (
                str(args.checkpoint.expanduser()) if args.checkpoint else None
            ),
            "infiller_weight": (
                str(args.infiller_weight.expanduser()) if args.infiller_weight else None
            ),
            "img_focal": args.img_focal,
            "static_camera": args.static_camera,
            "vis": args.vis,
            "max_frames": args.max_frames,
        },
        "software": {},
    }
