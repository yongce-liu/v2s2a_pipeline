"""Tests for the temporal post-pass (jump gating, bridging, holds), no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pose_estimation.temporal import (
    JumpGateConfig,
    TrackFrame,
    fuse_track,
    gate_steps,
    load_poses,
    pose_stability_metrics,
    rotation_log,
    se3_distance,
    se3_slerp,
    segment_gaps,
    track_metrics,
)


def make_pose(tx: float = 0.0, angle_deg: float = 0.0) -> np.ndarray:
    """Pose rotating about z by ``angle_deg`` and translated on x by ``tx``."""

    a = np.radians(angle_deg)
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )
    pose[:3, 3] = [tx, 0.0, 0.5]
    return pose


def make_frames(poses: dict[int, np.ndarray], n: int) -> list[TrackFrame]:
    """Build a track where every frame is tracked, backward for the first half."""

    return [
        TrackFrame(
            index=i,
            pose=poses.get(i, make_pose()),
            tracked=i in poses,
            method="track",
            direction="backward" if i < n // 2 else "forward",
        )
        for i in range(n)
    ]


# ------------------------------------------------------------------ SO(3)/SE(3)


def test_rotation_log_roundtrip_identity() -> None:
    assert np.allclose(rotation_log(np.eye(3)), np.zeros(3))


def test_rotation_log_quarter_turn() -> None:
    R = make_pose(angle_deg=90.0)[:3, :3]
    w = rotation_log(R)
    assert np.isclose(np.linalg.norm(w), np.pi / 2)
    assert np.allclose(w / np.linalg.norm(w), [0.0, 0.0, 1.0])


def test_rotation_log_half_turn_axis() -> None:
    R = np.diag([1.0, -1.0, -1.0])  # 180 deg about x
    w = rotation_log(R)
    assert np.isclose(np.linalg.norm(w), np.pi)
    assert np.allclose(np.abs(w / np.linalg.norm(w)), [1.0, 0.0, 0.0])


def test_se3_distance_components() -> None:
    dt, dr = se3_distance(make_pose(0.0, 0.0), make_pose(0.03, 30.0))
    assert np.isclose(dt, 0.03)
    assert np.isclose(dr, 30.0)


def test_se3_slerp_endpoints_and_midpoint() -> None:
    Ta, Tb = make_pose(0.0, 0.0), make_pose(0.2, 60.0)
    assert np.allclose(se3_slerp(Ta, Tb, 0.0), Ta)
    assert np.allclose(se3_slerp(Ta, Tb, 1.0), Tb)
    mid = se3_slerp(Ta, Tb, 0.5)
    dt, dr = se3_distance(Ta, mid)
    assert np.isclose(dr, 30.0)
    assert np.isclose(dt, 0.1)
    # geodesic rotation preserves orthonormality
    assert np.allclose(mid[:3, :3] @ mid[:3, :3].T, np.eye(3))


def test_se3_slerp_shortest_path_across_wraparound() -> None:
    # 170 deg -> -170 deg must go the 20-degree short way, not 340.
    Ta, Tb = make_pose(angle_deg=170.0), make_pose(angle_deg=-170.0)
    _, dr = se3_distance(Ta, se3_slerp(Ta, Tb, 0.5))
    assert np.isclose(dr, 10.0)


# ------------------------------------------------------------------ gating


def test_gate_flags_only_the_outlier_step() -> None:
    poses = {i: make_pose(i * 0.001, i * 0.5) for i in range(20)}
    poses[10] = make_pose(0.01, 85.0)  # one-frame blow-out
    frames = make_frames(poses, 20)
    stats, dr_gate, dt_gate = gate_steps(frames, JumpGateConfig())
    flagged = [s.index for s in stats if s.is_jump]
    # both boundary steps of a one-frame blow-out are flagged: the step into
    # the outlier (10) and the step back out of it (11)
    assert flagged == [10, 11]
    assert dr_gate >= 20.0
    assert dt_gate >= 0.06


def test_gate_healthy_slow_motion_not_flagged() -> None:
    # Steady, plausible motion: 1.5 deg and 5 mm per frame throughout.
    poses = {i: make_pose(i * 0.005, i * 1.5) for i in range(30)}
    stats, _, _ = gate_steps(make_frames(poses, 30), JumpGateConfig())
    assert not any(s.is_jump for s in stats)


def test_segment_gaps_contiguous_and_bounded() -> None:
    poses = {i: make_pose(0.0, 0.0) for i in range(20)}
    poses[5] = make_pose(0.0, 80.0)
    poses[6] = make_pose(0.0, 85.0)
    poses[12] = make_pose(0.0, -70.0)
    stats, _, _ = gate_steps(make_frames(poses, 20), JumpGateConfig())
    spans = segment_gaps(stats)
    # first excursion: 4->5 (jump), 5->6 (moving 5deg), 6->7 (jump back) merge;
    # second: 11->12 and 12->13 (two consecutive jumps) merge.
    assert spans == [(5, 7), (12, 13)]


def test_gate_falls_back_to_floors_with_no_healthy_leg() -> None:
    # All-forward track (<8 backward steps): absolute floors must apply.
    poses = {i: make_pose(0.0, i * 1.0) for i in range(6)}
    poses[5] = make_pose(0.0, 95.0)
    frames = [TrackFrame(i, poses[i], True, "track", "forward") for i in range(6)]
    stats, dr_gate, dt_gate = gate_steps(frames, JumpGateConfig())
    assert dr_gate == 20.0
    assert dt_gate == 0.06
    assert any(s.is_jump for s in stats)


# ------------------------------------------------------------------ fusion


def test_fuse_bridges_jump_span_and_keeps_healthy_frames() -> None:
    poses = {i: make_pose(0.001 * i, 0.5 * i) for i in range(12)}
    for i in (5, 6, 7):  # corrupted span
        poses[i] = make_pose(0.001 * i, 120.0)
    frames = make_frames(poses, 12)
    fused, verdicts = fuse_track(frames, JumpGateConfig())
    by_index = {f.index: f for f in fused}
    actions = {v.frame_index: v.action for v in verdicts}

    # Healthy frames are byte-identical to the measurement.
    for i in (0, 1, 2, 3, 9, 10, 11):
        assert actions[i] == "keep"
        assert np.allclose(by_index[i].pose, poses[i])

    # The corruption's boundary steps (4-5 and 7-8) exceed the gate, while the
    # frozen middle reads as "holding still", so it splits into two excursions;
    # both get bridged to their nearest healthy pivots instead of the 120-deg
    # measurement.
    assert actions[4] == actions[5] == "bridge"
    assert actions[6] == "keep"
    assert actions[7] == actions[8] == "bridge"
    for i in (4, 5, 7, 8):
        _, dr = se3_distance(by_index[i].pose, make_pose(0.001 * i, 120.0))
        assert dr > 20.0  # bridged poses are far from the corrupted rotation


def test_fuse_hold_when_span_runs_to_clip_end() -> None:
    poses = {i: make_pose(0.001 * i, 0.5 * i) for i in range(10)}
    for i in (8, 9):  # tracker lost it at the end of the clip
        poses[i] = make_pose(0.001 * i, 130.0)
    fused, verdicts = fuse_track(make_frames(poses, 10), JumpGateConfig())
    by_index = {f.index: f for f in fused}
    actions = {v.frame_index: v.action for v in verdicts}
    assert actions[7] == actions[8] == "hold"
    assert np.allclose(by_index[7].pose, poses[6])
    assert np.allclose(by_index[8].pose, poses[6])
    assert actions[9] == "keep"
    assert np.allclose(by_index[9].pose, poses[9])


def test_fuse_single_frame_spike_is_bridged_back_to_line() -> None:
    poses = {i: make_pose(0.0, 0.0) for i in range(10)}
    poses[5] = make_pose(0.0, 90.0)  # single bad frame, reacquired right after
    fused, verdicts = fuse_track(make_frames(poses, 10), JumpGateConfig())
    actions = {v.frame_index: v.action for v in verdicts}
    by_index = {f.index: f for f in fused}
    # excursion covers poses 4..6 (untouched-pose window), bridged 3 -> 7
    assert actions[4] == actions[5] == actions[6] == "bridge"
    assert all(actions[i] == "keep" for i in (0, 1, 2, 3, 7, 8, 9))
    # the bridged frame sits on the interpolation chord, not at 90 deg
    _, dr = se3_distance(by_index[5].pose, make_pose(0.0, 0.0))
    assert dr < 10.0


def test_fuse_never_worsens_innovations_on_yellow_spoon_like_data() -> None:
    """Synthetic reproduction of the clip: slow lift, then end-of-clip rotas."""
    poses: dict[int, np.ndarray] = {}
    for i in range(89):
        poses[i] = make_pose(0.0001 * i, 0.2 * i)
    for k, i in enumerate(range(77, 89)):  # progressive spin drift to the end
        poses[i] = make_pose(0.0001 * i, 0.2 * i + 8.0 * (k + 1) * 5)
    frames = make_frames(poses, 89)
    before = [se3_distance(a.pose, b.pose) for a, b in zip(frames[:-1], frames[1:])]
    fused, _ = fuse_track(frames, JumpGateConfig())
    after = [se3_distance(a.pose, b.pose) for a, b in zip(fused[:-1], fused[1:])]
    assert max(dr for _, dr in after) <= max(dr for _, dr in before)
    assert sum(1 for _, dr in after if dr > 20) == 0


def test_load_poses_roundtrip(tmp_path: Path) -> None:
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    np.savetxt(poses_dir / "000000.txt", make_pose(0.1, 10.0))
    manifest = {
        "poses_dir": str(poses_dir),
        "entries": [
            {
                "index": 1,
                "pose_filename": "000000.txt",
                "tracked": True,
                "method": "track",
                "direction": "forward",
            },
            {
                "index": 0,
                "pose_filename": None,
                "tracked": False,
                "method": "skipped",
                "direction": "skipped",
            },
        ],
    }
    path = tmp_path / "poses.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _, frames = load_poses(path)
    assert [f.index for f in frames] == [0, 1]
    assert frames[0].pose is None
    assert frames[1].pose is not None
    assert np.isclose(se3_distance(frames[1].pose, make_pose(0.1, 10.0))[1], 0.0)


# ------------------------------------------------------------------ metrics


def test_pose_stability_static_tail() -> None:
    poses = {i: make_pose(0.5, 10.0) for i in range(20)}  # perfectly static
    stability = pose_stability_metrics(make_frames(poses, 20))
    assert stability["max_dt_from_window_m"] == pytest.approx(0.0)
    assert stability["max_dr_from_window_deg"] == pytest.approx(0.0)
    assert stability["frac_within_tol"] == pytest.approx(1.0)


def test_track_metrics_counts_large_steps() -> None:
    poses = {i: make_pose(0.0, 0.0) for i in range(6)}
    poses[3] = make_pose(0.0, 45.0)
    metrics = track_metrics(make_frames(poses, 6))
    assert metrics["summary"]["steps_over_20deg"] >= 2  # in and out of the spike
    assert metrics["summary"]["dr_deg"]["max"] >= 44.0
