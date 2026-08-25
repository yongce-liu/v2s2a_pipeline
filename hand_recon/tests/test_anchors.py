"""Tests for the published HaWoR hand-anchor artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hand_recon.anchors import build_hand_anchors_payload


def test_build_hand_anchors_payload_is_frame_aligned(tmp_path: Path) -> None:
    tracks_path = tmp_path / "model_tracks.npy"
    np.save(
        tracks_path,
        {
            1.0: [
                {
                    "frame": 0,
                    "det": True,
                    "det_box": np.asarray([[10, 20, 30, 40, 0.8]], dtype=np.float32),
                    "det_handedness": np.asarray([1], dtype=np.float32),
                }
            ],
            2.0: [
                {
                    "frame": 1,
                    "det": True,
                    "det_box": np.asarray([[1, 2, 9, 10, 0.7]], dtype=np.float32),
                    "det_handedness": np.asarray([0], dtype=np.float32),
                }
            ],
        },
        allow_pickle=True,
    )
    manifest = {
        "source_frames_json": "/tmp/frames.json",
        "source_video": "/tmp/video.mp4",
        "fps": 30.0,
        "width": 100,
        "height": 80,
        "frame_count": 2,
        "entries": [
            {"index": 0, "frame_filename": "000000.png", "timestamp_sec": 0.0},
            {"index": 1, "frame_filename": "000001.png", "timestamp_sec": 1 / 30},
        ],
    }

    payload = build_hand_anchors_payload(tracks_path, manifest, 0, 2)

    assert payload["schema_version"] == "1.0"
    assert payload["coordinate_space"] == "pixel"
    assert payload["box_format"] == "xyxy"
    assert payload["targets"] == [
        {"prompt_id": "left hand", "anchor_type": "box"},
        {"prompt_id": "right hand", "anchor_type": "box"},
    ]
    assert payload["processed_count"] == 2
    assert payload["entries"][0]["frame_filename"] == "000000.png"
    assert payload["entries"][0]["anchors"]["left hand"] is None
    right = payload["entries"][0]["anchors"]["right hand"]
    assert right == {
        "side": "right",
        "track_id": 1,
        "box_xyxy": [10.0, 20.0, 30.0, 40.0],
        "point_xy": [20.0, 30.0],
        "confidence": pytest.approx(0.8),
        "handedness_score": 1.0,
    }
    left = payload["entries"][1]["anchors"]["left hand"]
    assert left["track_id"] == 2
    assert left["point_xy"] == [5.0, 6.0]
