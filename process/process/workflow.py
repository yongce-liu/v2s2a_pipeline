"""End-to-end ingestion: probe, extract, and write stage outputs."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from process import __version__
from process.extract import ExtractArgs, ExtractOutputs, extract_frames
from process.probe import VideoInfo, probe_video


@dataclass(frozen=True)
class StageRun:
    """Everything produced by one ingestion run for one source video."""

    clip_root: Path
    stage_dir: Path
    frames_dir: Path
    video_info_path: Path
    frames_json_path: Path
    config_json_path: Path
    video_info: VideoInfo
    manifest: ExtractOutputs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_ingest(
    video_path: Path,
    output_root: Path,
    extract_args: ExtractArgs,
) -> StageRun:
    """Run the full ingestion stage for one source video.

    ``video_path`` is treated as read-only. Outputs are written to
    ``<output_root>/<video_stem>/process/`` and existing outputs are
    preserved unless ``extract_args.overwrite`` is set. Idempotent: a repeated
    run with unchanged inputs and config reuses prior outputs.
    """
    video_path = video_path.expanduser().resolve()

    result = probe_video(video_path)
    if not result.ok:
        raise RuntimeError(result.error or "probe failed")

    clip_root = output_root / video_path.stem
    stage_dir = clip_root / "process"

    manifest_out = extract_frames(
        video_path=video_path,
        output_root=clip_root,
        args=extract_args,
        source_fps=result.info.fps,
        source_width=result.info.width,
        source_height=result.info.height,
    )

    config = _config_dict(extract_args, result.info)
    _write_json(stage_dir / "config.json", config)
    _write_json(stage_dir / "video_info.json", result.info.to_dict())
    _write_json(stage_dir / "frames.json", manifest_out.manifest.to_dict())

    logger.info(
        "[process] Done: frames={} fps={} size={}x{} out={}",
        manifest_out.manifest.frame_count,
        manifest_out.manifest.fps,
        manifest_out.manifest.width,
        manifest_out.manifest.height,
        stage_dir,
    )

    return StageRun(
        clip_root=clip_root,
        stage_dir=stage_dir,
        frames_dir=manifest_out.frames_dir,
        video_info_path=stage_dir / "video_info.json",
        frames_json_path=stage_dir / "frames.json",
        config_json_path=stage_dir / "config.json",
        video_info=result.info,
        manifest=manifest_out,
    )


def _config_dict(extract_args: ExtractArgs, info: VideoInfo) -> dict:
    """The effective config: inputs that define this run plus the source info."""

    return {
        "package": {"name": "process", "version": __version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_video": {
            "format": info.format,
            "container": info.container,
            "duration_sec": info.duration_sec,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "video_codec": info.video_codec,
            "audio_codec": info.audio_codec,
            "has_audio": info.has_audio,
        },
        "extract": {
            "fps": extract_args.fps,
            "width": extract_args.width,
            "height": extract_args.height,
            "format": extract_args.format,
            "overwrite": extract_args.overwrite,
        },
        "software": {
            "ffprobe_version": info.ffprobe_version,
            "ffmpeg_version": _ffmpeg_version(),
        },
    }


def _ffmpeg_version() -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return "not-found"
    proc = subprocess.run(
        [ffmpeg_bin, "-version"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.splitlines()[0] if proc.stdout else "unknown"
