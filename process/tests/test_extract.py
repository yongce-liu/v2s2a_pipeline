"""Tests for ffmpeg-based frame extraction."""

from __future__ import annotations

import itertools
import shutil
import subprocess
from pathlib import Path

import pytest

from process.extract import ExtractArgs, extract_frames


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny 0.5 s, 10 fps synthetic video."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")

    path = tmp_path_factory.mktemp("extract") / "sample.mp4"
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=0.5",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg failed to create sample video: {proc.stderr}")
    return path


def test_extract_frames(sample_video: Path, tmp_path: Path) -> None:
    output = extract_frames(
        video_path=sample_video,
        output_root=tmp_path,
        args=ExtractArgs(fps=10),
        source_fps=10.0,
        source_width=64,
        source_height=48,
    )

    manifest = output.manifest
    assert manifest.frame_count >= 5  # 0.5 s at 10 fps
    assert output.frames_dir.exists()
    frames = sorted(output.frames_dir.glob("*.png"))
    assert len(frames) == manifest.frame_count
    assert frames[0].name == "000000.png"
    # Manifest timestamps are strictly monotonic at 1/fps spacing.
    stamps = [entry.timestamp_sec for entry in manifest.entries]
    assert all(s is not None for s in stamps)
    assert all(b - a == pytest.approx(0.1) for a, b in itertools.pairwise(stamps))


def test_extract_preserves_existing(tmp_path: Path) -> None:
    frames_dir = tmp_path / "process" / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "000000.png").write_bytes(b"placeholder")

    output = extract_frames(
        video_path=tmp_path / "whatever.mp4",  # path never touched
        output_root=tmp_path,
        args=ExtractArgs(fps=10),
        source_fps=None,
        source_width=None,
        source_height=None,
    )
    assert output.manifest.frame_count == 1


def test_extract_overwrite_clears(sample_video: Path, tmp_path: Path) -> None:
    first = extract_frames(
        video_path=sample_video,
        output_root=tmp_path,
        args=ExtractArgs(fps=10),
        source_fps=10.0,
        source_width=64,
        source_height=48,
    )
    marker = first.frames_dir / "stale_placeholder.png"
    marker.write_bytes(b"placeholder")

    second = extract_frames(
        video_path=sample_video,
        output_root=tmp_path,
        args=ExtractArgs(fps=10, overwrite=True),
        source_fps=10.0,
        source_width=64,
        source_height=48,
    )
    assert not marker.exists()  # stale file cleared on overwrite
    assert second.manifest.frame_count >= 5
    assert sorted(second.frames_dir.glob("*.png")) != []


def test_extract_scale_expression() -> None:
    from process.extract import _resolve_output_size

    assert _resolve_output_size(None, None, 64, 48) is None
    assert _resolve_output_size(128, None, 64, 48) == "128:-2"
    assert _resolve_output_size(None, 96, 64, 48) == "-2:96"
    assert _resolve_output_size(64, 48, 64, 48) is None  # unchanged source size
    assert _resolve_output_size(128, 96, 64, 48) == "128:96"
