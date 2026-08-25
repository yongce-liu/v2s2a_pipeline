"""Publish HaWoR handedness detections as a stable pipeline artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

HAND_ANCHORS_FILENAME = "hand_anchors.json"


def _track_id_value(track_id: Any) -> int | float | str:
    if isinstance(track_id, np.generic):
        track_id = track_id.item()
    if isinstance(track_id, float) and track_id.is_integer():
        return int(track_id)
    if isinstance(track_id, (int, float, str)):
        return track_id
    return str(track_id)


def _collect_anchors(tracks_path: Path) -> dict[int, dict[str, dict]]:
    raw_tracks = np.load(tracks_path, allow_pickle=True).item()
    if not isinstance(raw_tracks, dict):
        raise TypeError(f"Invalid HaWoR tracks payload: {tracks_path}")

    by_frame: dict[int, dict[str, dict]] = {}
    for raw_track_id, observations in raw_tracks.items():
        track_id = _track_id_value(raw_track_id)
        for observation in observations:
            if not observation.get("det", False):
                continue
            handedness_score = float(
                np.asarray(observation["det_handedness"]).reshape(-1)[0]
            )
            side = "right" if handedness_score > 0.5 else "left"
            prompt_id = f"{side} hand"
            frame_index = int(observation["frame"])
            values = np.asarray(observation["det_box"], dtype=np.float32).reshape(-1)
            if values.size < 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in values[:4])
            confidence = float(values[4]) if values.size >= 5 else 1.0
            if x2 <= x1 or y2 <= y1:
                continue

            anchor = {
                "side": side,
                "track_id": track_id,
                "box_xyxy": [x1, y1, x2, y2],
                "point_xy": [(x1 + x2) / 2, (y1 + y2) / 2],
                "confidence": confidence,
                "handedness_score": handedness_score,
            }
            frame_anchors = by_frame.setdefault(frame_index, {})
            previous = frame_anchors.get(prompt_id)
            if previous is None or confidence > previous["confidence"]:
                frame_anchors[prompt_id] = anchor
    return by_frame


def build_hand_anchors_payload(
    tracks_path: Path,
    manifest: dict,
    start_index: int,
    end_index: int,
) -> dict:
    """Convert HaWoR's internal track pickle into stable, frame-aligned JSON."""

    tracks_path = tracks_path.expanduser().resolve()
    if not tracks_path.is_file():
        raise FileNotFoundError(f"HaWoR model tracks not found: {tracks_path}")
    if end_index < start_index:
        raise ValueError("end_index must be >= start_index.")

    by_frame = _collect_anchors(tracks_path)
    manifest_entries = {
        int(entry["index"]): entry for entry in manifest.get("entries", [])
    }
    entries = []
    for frame_index in range(start_index, end_index):
        source = manifest_entries.get(frame_index, {})
        entries.append(
            {
                "index": frame_index,
                "frame_filename": source.get("frame_filename"),
                "timestamp_sec": source.get("timestamp_sec"),
                "anchors": {
                    "left hand": by_frame.get(frame_index, {}).get("left hand"),
                    "right hand": by_frame.get(frame_index, {}).get("right hand"),
                },
            }
        )

    return {
        "schema_version": "1.0",
        "stage": "hand_recon",
        "source_frames_json": manifest.get("source_frames_json"),
        "source_video": manifest.get("source_video"),
        "fps": manifest.get("fps"),
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "frame_count": manifest.get("frame_count"),
        "processed_count": len(entries),
        "coordinate_space": "pixel",
        "box_format": "xyxy",
        "point_definition": "box_center",
        "generator": "hawor_detector",
        "targets": [
            {"prompt_id": "left hand", "anchor_type": "box"},
            {"prompt_id": "right hand", "anchor_type": "box"},
        ],
        "entries": entries,
    }
