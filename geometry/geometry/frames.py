"""Loading the ``process`` stage frame manifest (``frames.json``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameEntry:
    """One frame from the process manifest."""

    index: int
    frame_filename: str
    timestamp_sec: float | None
    path: Path
    """Absolute path to the frame image on disk."""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frame_filename": self.frame_filename,
            "timestamp_sec": self.timestamp_sec,
        }


@dataclass(frozen=True)
class FrameManifest:
    """Parsed ``frames.json`` written by the ``process`` stage."""

    source_video: str
    fps: float | None
    width: int | None
    height: int | None
    format: str
    frame_count: int
    frames_dir: Path
    entries: list[FrameEntry]

    def to_dict(self) -> dict:
        return {
            "source_video": self.source_video,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "frame_count": self.frame_count,
            "frames_dir": str(self.frames_dir),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def load_frame_manifest(frames_json: Path) -> FrameManifest:
    """Read and validate a ``process`` ``frames.json`` file.

    The frame images are resolved against the ``frames_dir`` recorded in the
    manifest, so downstream stages never guess filenames.
    """
    frames_json = frames_json.expanduser()
    if not frames_json.exists():
        raise FileNotFoundError(f"frames.json not found: {frames_json}")

    data = json.loads(frames_json.read_text(encoding="utf-8"))
    frames_dir = Path(data.get("frames_dir", "")).expanduser()
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_dir from frames.json not found: {frames_dir}")

    entries: list[FrameEntry] = []
    for raw in data.get("entries", []):
        index = int(raw["index"])
        frame_filename = raw["frame_filename"]
        entries.append(
            FrameEntry(
                index=index,
                frame_filename=frame_filename,
                timestamp_sec=raw.get("timestamp_sec"),
                path=(frames_dir / frame_filename).resolve(),
            )
        )
    if not entries:
        raise ValueError(f"No frame entries in frames.json: {frames_json}")

    return FrameManifest(
        source_video=data.get("source_video", ""),
        fps=data.get("fps"),
        width=data.get("width"),
        height=data.get("height"),
        format=data.get("format", "png"),
        frame_count=int(data.get("frame_count", len(entries))),
        frames_dir=frames_dir.resolve(),
        entries=entries,
    )
