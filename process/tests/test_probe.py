"""Tests for ffprobe-based video probing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from process.probe import _as_fps, probe_video


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic video, created once per test session via ffmpeg."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")

    path = tmp_path_factory.mktemp("probe") / "sample.mp4"
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


def test_probe_success(sample_video: Path) -> None:
    result = probe_video(sample_video)
    assert result.ok, result.error
    assert result.info is not None
    info = result.info
    assert info.has_video
    assert info.width == 64
    assert info.height == 48
    assert info.frame_count is None or info.frame_count >= 1
    assert info.probe  # raw ffprobe JSON retained


def test_probe_missing_file(tmp_path: Path) -> None:
    result = probe_video(tmp_path / "nope.mp4")
    assert not result.ok
    assert "not found" in (result.error or "")


def test_probe_non_video_file(tmp_path: Path) -> None:
    text = tmp_path / "not_a_video.txt"
    text.write_text("hello", encoding="utf-8")
    result = probe_video(text)
    if result.ok:
        # ffmpeg may still report the file as a valid stream container; probe
        # must never claim a video stream for a plain-text file.
        assert result.info is None or not result.info.has_video
    else:
        assert result.error


def test_as_fps_rational() -> None:
    assert _as_fps("30/1") == 30.0
    assert _as_fps("2997/100") == pytest.approx(29.97)
    assert _as_fps("29.97") == pytest.approx(29.97)
    assert _as_fps("1/0") is None
    assert _as_fps("bogus") is None
    assert _as_fps(None) is None
