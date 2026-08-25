"""Tests for the media helpers and the intrinsics conversion."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from geometry.media import colorize_depth, save_depth_exr, save_mask, save_points_ply
from geometry.moge_model import MogeFrameResult


def test_save_depth_exr_roundtrip(tmp_path: Path) -> None:
    depth = np.linspace(0.5, 3.0, 48, dtype=np.float32).reshape(6, 8)
    out = tmp_path / "depth.exr"
    save_depth_exr(depth, out, overwrite=True)

    loaded = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert loaded is not None
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded.reshape(depth.shape), depth, rtol=1e-5)


def test_save_mask_no_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "mask.png"
    save_mask(np.full((4, 4), 255, dtype=np.uint8), out, overwrite=True)
    first_mtime = out.stat().st_mtime_ns
    save_mask(np.zeros((4, 4), dtype=np.uint8), out, overwrite=False)
    assert out.stat().st_mtime_ns == first_mtime


def test_save_points_ply_drops_non_finite(tmp_path: Path) -> None:
    points = np.random.default_rng(0).normal(size=(4, 5, 3)) * 2.0 + 5.0
    points[1, 1] = np.inf
    colors = np.full((4, 5, 3), 200, dtype=np.uint8)
    out = tmp_path / "cloud.ply"

    save_points_ply(points, colors, out, overwrite=True)

    assert out.exists()
    header = out.read_bytes()[:512].decode("ascii", errors="ignore")
    assert header.startswith("ply")
    assert "format binary_little_endian" in header
    # 20 points minus the one non-finite vertex.
    assert "element vertex 19" in header


def test_colorize_depth_invalid_pixels_black() -> None:
    depth = np.array([[1.0, 2.0], [np.inf, -1.0]], dtype=np.float32)
    colored = colorize_depth(depth)
    assert colored.shape == (2, 2, 3)
    assert (colored[1, 0] == 0).all()
    assert (colored[1, 1] == 0).all()
    assert colored[0, 0].any()


def test_denormalized_intrinsics() -> None:
    result = MogeFrameResult(
        points=np.zeros((6, 8, 3)),
        depth=np.zeros((6, 8)),
        mask=None,
        normal=None,
        intrinsics=np.array([[0.5, 0.0, 0.25], [0.0, 0.5, 0.75], [0.0, 0.0, 1.0]]),
    )
    k = result.denormalized_intrinsics(width=640, height=480)
    assert k[0, 0] == pytest.approx(320.0)
    assert k[0, 2] == pytest.approx(160.0)
    assert k[1, 1] == pytest.approx(240.0)
    assert k[1, 2] == pytest.approx(360.0)
    assert k[2, 2] == pytest.approx(1.0)


def test_load_rgb_image(tmp_path: Path) -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200  # red left half of the RGB image
    image_path = tmp_path / "frame.png"
    Image.fromarray(rgb).save(image_path)

    from geometry.media import load_rgb_image

    loaded = load_rgb_image(image_path)
    assert loaded.shape == (4, 6, 3)
    assert loaded[0, 0, 0] == 200
