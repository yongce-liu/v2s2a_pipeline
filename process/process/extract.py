"""ffmpeg-based frame extraction for the process package."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Shared frame-manifest schema version (all stages write these common keys).
FRAME_MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FrameManifestEntry:
    """One extracted frame in the output ``frames`` directory."""

    index: int
    filename: str
    timestamp_sec: float | None
    """Decode timestamp in seconds, derived from ``-vf fps`` resampling; may be
    None when the output is written without timestamps."""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frame_filename": self.filename,
            "timestamp_sec": self.timestamp_sec,
        }


@dataclass(frozen=True)
class FrameManifest:
    """The full manifest of a successful frame extraction."""

    source_video: str
    fps: float | None
    width: int | None
    height: int | None
    format: str
    frame_count: int
    frames_dir: str
    ffmpeg_version: str
    entries: list[FrameManifestEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": FRAME_MANIFEST_SCHEMA_VERSION,
            "stage": "process",
            "source_video": self.source_video,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "frames_dir": self.frames_dir,
            "frame_format": self.format,
            "format": self.format,
            "entries": [entry.to_dict() for entry in self.entries],
            "ffmpeg_version": self.ffmpeg_version,
        }


@dataclass
class ExtractArgs:
    """Arguments controlling frame extraction."""

    fps: float | None = None
    """Uniform extraction rate in Hz. Omit (None) to keep every source frame."""
    width: int | None = None
    """Output width in pixels. Omit to keep the source width; when combined with
    ``height`` only one axis needs to be given (the other stays proportional)."""
    height: int | None = None
    """Output height in pixels. Omit to keep the source height."""
    format: str = "png"
    """Output image format: ``png`` or ``jpg``."""
    overwrite: bool = False
    """Clear an existing frames directory and re-extract. Without it, existing
    extracted frames are kept and re-extraction is skipped."""


@dataclass
class ExtractOutputs:
    """Outputs produced by ``extract_frames``."""

    frames_dir: Path
    manifest: FrameManifest


def _frame_pattern(format: str, zero_padding: int) -> str:
    if format == "jpg":
        return f"%0{zero_padding}d.jpg"
    return f"%0{zero_padding}d.png"


def _ffmpeg_version(ffmpeg_bin: str) -> str:
    proc = subprocess.run(
        [ffmpeg_bin, "-version"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.splitlines()[0] if proc.stdout else "unknown"


def _resolve_output_size(
    width: int | None,
    height: int | None,
    source_width: int | None,
    source_height: int | None,
) -> str | None:
    """Resolve the ffmpeg scale expression.

    ``-2`` tells ffmpeg to derive the omitted axis proportionally. ``iw/ih``
    (and ``-iw/-ih``) preserve the source extent so that providing no size
    leaves the frame untouched.
    """
    if width is None and height is None:
        return None

    if width is None:
        return f"-2:{height}"
    if height is None:
        return f"{width}:-2"

    if source_width == width and source_height == height:
        return None
    return f"{width}:{height}"


def extract_frames(
    video_path: Path,
    output_root: Path,
    args: ExtractArgs,
    source_fps: float | None,
    source_width: int | None,
    source_height: int | None,
) -> ExtractOutputs:
    """Extract frames from ``video_path`` into ``output_root``.

    The output path layout is ``<output_root>/process/frames``; the caller
    decides whether ``output_root`` already carries the clip stem. Existing
    outputs are preserved unless ``args.overwrite`` is set.
    """
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--extract.fps must be positive.")
    if args.width is not None and args.width <= 0:
        raise ValueError("--extract.width must be positive.")
    if args.height is not None and args.height <= 0:
        raise ValueError("--extract.height must be positive.")
    if args.format not in ("png", "jpg"):
        raise ValueError("--extract.format must be 'png' or 'jpg'.")

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise FileNotFoundError("ffmpeg executable not found on PATH")

    stage_dir = output_root / "process"
    frames_dir = stage_dir / "frames"
    if frames_dir.exists() and not args.overwrite:
        existing = sorted(frames_dir.glob(f"*.{args.format}"))
        if existing:
            entries = [
                FrameManifestEntry(
                    index=int(path.stem),
                    filename=path.name,
                    timestamp_sec=None,
                )
                for path in existing
            ]
            manifest = FrameManifest(
                source_video=str(video_path),
                fps=args.fps,
                width=args.width,
                height=args.height,
                format=args.format,
                frame_count=len(entries),
                frames_dir=str(frames_dir),
                entries=entries,
                ffmpeg_version=_ffmpeg_version(ffmpeg_bin),
            )
            return ExtractOutputs(frames_dir=frames_dir, manifest=manifest)

    if frames_dir.exists() and args.overwrite:
        shutil.rmtree(frames_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)

    width, height = args.width, args.height
    if width is None and height is None:
        width, height = source_width, source_height

    scale = _resolve_output_size(args.width, args.height, source_width, source_height)

    vf = []
    if args.fps is not None:
        vf.append(f"fps={args.fps}")
    if scale is not None:
        vf.append(f"scale={scale}")
    vf_arg = ",".join(vf) if vf else None

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
    ]
    if vf_arg:
        cmd += ["-vf", vf_arg]
    cmd += [
        "-f",
        "image2",
        "-start_number",
        "0",
        str(frames_dir / _frame_pattern(args.format, 6)),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed for {video_path} (rc={proc.returncode}): "
            f"{proc.stderr.strip()}"
        )

    frames = sorted(frames_dir.glob(f"*.{args.format}"))
    fps = args.fps or source_fps
    entries = []
    for index, path in enumerate(frames):
        timestamp = (index / fps) if (fps is not None and fps > 0) else None
        entries.append(
            FrameManifestEntry(
                index=index,
                filename=path.name,
                timestamp_sec=timestamp,
            )
        )

    manifest = FrameManifest(
        source_video=str(video_path),
        fps=fps,
        width=width,
        height=height,
        format=args.format,
        frame_count=len(entries),
        frames_dir=str(frames_dir),
        entries=entries,
        ffmpeg_version=_ffmpeg_version(ffmpeg_bin),
    )
    return ExtractOutputs(frames_dir=frames_dir, manifest=manifest)
