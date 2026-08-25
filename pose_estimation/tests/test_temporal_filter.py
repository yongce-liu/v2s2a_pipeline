"""CPU-only tests for the late-frame pose temporal filter.

No GPU, FoundationPose, trimesh, or manifest files are exercised here: the
filter and its noise-scaling helpers are pure numpy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pose_estimation.temporal_filter import (
    PoseTemporalFilter,
    TemporalFilterArgs,
    apply_temporal_filter_to_run,
    measurement_noise_scale,
    rotation_exp,
    rotation_log,
)


def _pose(t, rv=None):
    T = np.eye(4)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    if rv is not None:
        T[:3, :3] = rotation_exp(np.asarray(rv, dtype=np.float64))
    return T


# ---------------------------------------------------------------------------
# SO(3) helpers.
# ---------------------------------------------------------------------------


def test_rotation_log_exp_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        rv = rng.normal(size=3) * rng.uniform(0.1, 2.5)
        R = rotation_exp(rv)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10
        assert np.allclose(rotation_log(R), rv, atol=1e-8)


def test_rotation_log_near_pi() -> None:
    rv = np.array([np.pi, 0.0, 0.0])
    back = rotation_log(rotation_exp(rv))
    assert np.allclose(np.linalg.norm(back), np.pi, atol=1e-6)


# ---------------------------------------------------------------------------
# EKF behavior.
# ---------------------------------------------------------------------------


def test_constant_velocity_passes_through() -> None:
    """A noise-free constant-velocity trajectory stays essentially unchanged."""

    args = TemporalFilterArgs()
    filt = PoseTemporalFilter(args, mesh_extent_m=0.15)
    N, dt = 40, 1.0 / 30.0
    v = np.array([0.5, -0.1, 0.2])
    poses = [_pose(v * (k * dt), rv=[0.0, 0.0, 0.05 * k]) for k in range(N)]
    smoothed, stats = filt.run(poses, [1.0] * N, dt)
    assert len(smoothed) == N
    assert not any(s.gated for s in stats)
    for raw, sm in zip(poses[2:], smoothed[2:]):
        assert np.allclose(sm[:3, 3], raw[:3, 3], atol=7e-3)


def test_outlier_is_gated_and_coast_recovers() -> None:
    """A large symmetric flip is rejected; the next on-trajectory frame accepted."""

    args = TemporalFilterArgs()
    filt = PoseTemporalFilter(args, mesh_extent_m=0.15)
    N, dt = 30, 1.0 / 30.0
    v = np.array([0.3, 0.0, 0.0])
    poses = [_pose(v * (k * dt)) for k in range(N)]
    # Inject a 90 deg flip at frame 15.
    flipped = poses[15].copy()
    flipped[:3, :3] = rotation_exp(np.array([0.0, 0.0, np.pi / 2]))
    poses[15] = flipped

    smoothed, stats = filt.run(poses, [1.0] * N, dt)
    assert stats[15].gated
    # The flip must not leak into the output pose.
    rot_err = np.linalg.norm(rotation_log(smoothed[15][:3, :3].T @ poses[14][:3, :3]))
    assert rot_err < np.deg2rad(5.0)
    # Tracking re-acquires on frame 16 (post-coast gate).
    assert not stats[16].gated
    assert stats[16].mahalanobis is not None


def test_missing_frames_coast_and_resume() -> None:
    """None poses are pure predictions; filtering resumes without re-init."""

    args = TemporalFilterArgs()
    filt = PoseTemporalFilter(args, mesh_extent_m=0.15)
    dt = 1.0 / 30.0
    v = np.array([0.2, 0.0, 0.0])
    poses: list = [_pose(v * (k * dt)) for k in range(20)]
    poses[8] = None
    poses[9] = None
    smoothed, stats = filt.run(poses, [1.0] * 20, dt)
    assert stats[8].coasted and stats[9].coasted
    # Coasted poses extrapolate the learned velocity.
    gap = smoothed[9][:3, 3] - smoothed[7][:3, 3]
    assert np.allclose(gap, 2 * v * dt, atol=5e-3)
    assert not stats[10].gated  # re-acquisition passes the coast gate


def test_noise_scaling_changes_gain() -> None:
    """High-noise measurements are trusted less (smaller correction)."""

    dt = 1.0 / 30.0
    base = [_pose([k * dt * 0.3, 0, 0]) for k in range(12)]
    perturbed = [p.copy() for p in base]
    perturbed[6][:3, 3] += np.array([0.0, 0.05, 0.0])  # 5 cm nudge

    args = TemporalFilterArgs()
    smooth_low, _ = PoseTemporalFilter(args).run(perturbed, [1.0] * 12, dt)
    smooth_high, _ = PoseTemporalFilter(args).run(perturbed, [10.0] * 12, dt)
    pull_low = abs(smooth_low[6][1, 3] - base[6][1, 3])
    pull_high = abs(smooth_high[6][1, 3] - base[6][1, 3])
    assert pull_low > pull_high > 0.0


def test_rts_velocity_smooths_without_moving_anchor() -> None:
    """The velocity-only RTS pass leaves accepted poses untouched."""

    args = TemporalFilterArgs()
    filt = PoseTemporalFilter(args, mesh_extent_m=0.15)
    dt = 1.0 / 30.0
    poses = [_pose([0.1 * k * dt * 30, 0.0, 0.0]) for k in range(10)]
    smoothed, _stats = filt.run(poses, [1.0] * 10, dt)
    # Anchor pose (frame 0) is exact.
    assert np.allclose(smoothed[0], poses[0], atol=0.0)


# ---------------------------------------------------------------------------
# measurement_noise_scale.
# ---------------------------------------------------------------------------


def test_noise_scale_components() -> None:
    args = TemporalFilterArgs()
    base = measurement_noise_scale(
        args,
        is_registration=False,
        mask_area_px=None,
        depth_valid_fraction=None,
        track_refine_iter=None,
    )
    assert base == pytest.approx(1.0)

    reg = measurement_noise_scale(
        args,
        is_registration=True,
        mask_area_px=None,
        depth_valid_fraction=None,
        track_refine_iter=None,
    )
    assert reg == pytest.approx(args.registration_quality_scale)

    half_iters = measurement_noise_scale(
        args,
        is_registration=False,
        mask_area_px=None,
        depth_valid_fraction=None,
        track_refine_iter=5,
    )
    assert half_iters == pytest.approx(np.sqrt(2.0), rel=1e-6)

    small_mask = measurement_noise_scale(
        args,
        is_registration=False,
        mask_area_px=int(args.mask_area_reference_px / 2),
        depth_valid_fraction=None,
        track_refine_iter=None,
    )
    assert small_mask == pytest.approx(2.0, rel=1e-6)

    tiny_mask = measurement_noise_scale(
        args,
        is_registration=False,
        mask_area_px=10,
        depth_valid_fraction=None,
        track_refine_iter=None,
    )
    assert tiny_mask == pytest.approx(1.0 / args.mask_area_min_quality)


# ---------------------------------------------------------------------------
# Manifest-driven entry point.
# ---------------------------------------------------------------------------


def test_apply_temporal_filter_to_run(tmp_path: Path) -> None:
    stage_dir = tmp_path / "pose_estimation"
    poses_dir = stage_dir / "poses"
    poses_dir.mkdir(parents=True)

    N, dt = 12, 1.0 / 30.0
    entries = []
    for k in range(N):
        fname = f"{k:06d}.txt"
        np.savetxt(poses_dir / fname, _pose([0.1 * k * dt * 30, 0.0, 0.0]))
        entries.append(
            {
                "index": k,
                "frame_filename": f"{k:06d}.png",
                "timestamp_sec": k * dt,
                "pose_filename": fname,
                "tracked": True,
                "method": "obj-recon-seed-refine" if k == 5 else "track",
                "direction": "forward",
                "anchor_frame": 5,
            }
        )
    manifest = {
        "schema_version": "2.0",
        "stage": "pose_estimation",
        "fps": 30.0,
        "entries": entries,
    }
    (stage_dir / "poses.json").write_text(json.dumps(manifest))

    args = TemporalFilterArgs(enabled=True)
    out_path = apply_temporal_filter_to_run(
        stage_dir, args, mesh_extent_m=0.15, track_refine_iter=10
    )
    assert out_path == stage_dir / "poses_filtered.json"
    out = json.loads(out_path.read_text())
    assert out["temporal_filter"]["gated_count"] == 0
    assert (stage_dir / "poses_filtered" / "000005.txt").exists()
    anchor = np.loadtxt(stage_dir / "poses_filtered" / "000005.txt").reshape(4, 4)
    raw = np.loadtxt(poses_dir / "000005.txt").reshape(4, 4)
    assert np.allclose(anchor, raw, atol=0.0)  # registration seed kept exact


def test_apply_temporal_filter_in_place(tmp_path: Path) -> None:
    stage_dir = tmp_path / "pose_estimation"
    poses_dir = stage_dir / "poses"
    poses_dir.mkdir(parents=True)
    np.savetxt(poses_dir / "000000.txt", np.eye(4))
    manifest = {
        "fps": 30.0,
        "entries": [
            {
                "index": 0,
                "frame_filename": "000000.png",
                "timestamp_sec": 0.0,
                "pose_filename": "000000.txt",
                "tracked": True,
                "method": "register",
                "direction": "register",
                "anchor_frame": 0,
            }
        ],
    }
    (stage_dir / "poses.json").write_text(json.dumps(manifest))
    args = TemporalFilterArgs(enabled=True, write_manifest=False)
    out_path = apply_temporal_filter_to_run(stage_dir, args)
    assert out_path == stage_dir / "poses.json"
    assert not (stage_dir / "poses_filtered.json").exists()


def test_missing_poses_json_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        apply_temporal_filter_to_run(
            tmp_path / "nope", TemporalFilterArgs(enabled=True)
        )
