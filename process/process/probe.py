"""ffprobe-based probing of video files for the process package."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    """Normalized metadata for a source video, extracted by ``ffprobe``."""

    format: str
    container: str
    duration_sec: float | None
    stream_count: int
    video_codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    audio_codec: str | None
    has_video: bool
    has_audio: bool
    ffprobe_version: str
    probe: dict = field(default_factory=dict)
    """Raw ffprobe JSON, retained for fields not normalized above."""

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "container": self.container,
            "duration_sec": self.duration_sec,
            "stream_count": self.stream_count,
            "video_codec": self.video_codec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "audio_codec": self.audio_codec,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "ffprobe_version": self.ffprobe_version,
        }


@dataclass(frozen=True)
class ProbeResult:
    """Probing outcome: either a valid ``VideoInfo`` or a diagnostic error."""

    ok: bool
    error: str | None = None
    info: VideoInfo | None = None
    # The stdout/stderr captured from the failed ffprobe invocation, for debugging.
    captured: str | None = None


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_fps(value: str | None) -> float | None:
    """Parse an ffprobe frame-rate field (e.g. ``"30/1"`` or ``"29.97"``)."""
    if value is None:
        return None
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    return _as_float(value)


def _find_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        raise FileNotFoundError("ffprobe executable not found on PATH")
    return path


def probe_video(video_path: Path, ffprobe: str = "ffprobe") -> ProbeResult:
    """Probe a video with ffprobe and return normalized metadata.

    Uses ``ffprobe -show_streams -show_format -of json``. Returns a
    ``ProbeResult.ok == False`` value instead of raising when the file is
    missing, unreadable, or has no video stream, so a CLI can report it as a
    per-task failure.
    """
    if not video_path.exists():
        return ProbeResult(False, f"Input video not found: {video_path}")

    ffprobe_bin = shutil.which(ffprobe)
    if ffprobe_bin is None:
        return ProbeResult(False, f"ffprobe executable not found: {ffprobe}")

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return ProbeResult(
            False,
            f"ffprobe failed for {video_path} (rc={proc.returncode})",
            captured=proc.stderr or proc.stdout,
        )
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return ProbeResult(False, f"Invalid ffprobe JSON for {video_path}: {exc}")

    streams = raw.get("streams", [])
    fmt = raw.get("format", {})
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        return ProbeResult(
            False,
            f"No video stream found in {video_path}",
            captured=proc.stdout,
        )

    first_video = video_streams[0]
    fps = _as_fps(first_video.get("avg_frame_rate"))
    info = VideoInfo(
        format=fmt.get("format_name", "unknown"),
        container=fmt.get("format_long_name", "unknown"),
        duration_sec=_as_float(fmt.get("duration")),
        stream_count=len(streams),
        video_codec=first_video.get("codec_name"),
        width=int(first_video["width"]) if first_video.get("width") else None,
        height=int(first_video["height"]) if first_video.get("height") else None,
        fps=fps,
        frame_count=int(first_video["nb_frames"])
        if first_video.get("nb_frames")
        else None,
        audio_codec=audio_streams[0].get("codec_name") if audio_streams else None,
        has_video=True,
        has_audio=bool(audio_streams),
        ffprobe_version=_ffprobe_version(ffprobe_bin),
        probe=raw,
    )
    return ProbeResult(True, info=info)


def _ffprobe_version(ffprobe_bin: str) -> str:
    proc = subprocess.run(
        [ffprobe_bin, "-version"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.splitlines()[0] if proc.stdout else "unknown"
