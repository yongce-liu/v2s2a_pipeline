"""Tests for pose_estimation media helpers (intrinsics / mask loading)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pose_estimation.media import load_intrinsics, load_mask, load_rgb_image


def test_load_rgb_image(tmp_path: Path) -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # red channel set
    Image.fromarray(rgb).save(tmp_path / "frame.png")

    loaded = load_rgb_image(tmp_path / "frame.png")
    assert loaded.shape == (4, 6, 3)
    assert loaded[0, 0, 0] == 255


def test_load_rgb_image_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rgb_image(tmp_path / "missing.png")


def test_load_mask(tmp_path: Path) -> None:
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")

    loaded = load_mask(tmp_path / "mask.png")
    assert loaded.dtype == np.bool_
    assert loaded.sum() == 4


def test_load_intrinsics(tmp_path: Path) -> None:
    intrinsics = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    np.save(tmp_path / "intrinsics.npy", intrinsics)

    loaded = load_intrinsics(tmp_path / "intrinsics.npy")
    assert loaded.shape == (3, 3)
    assert loaded[0, 0] == pytest.approx(500.0)


def test_load_intrinsics_bad_shape(tmp_path: Path) -> None:
    np.save(tmp_path / "intrinsics.npy", np.eye(4))
    with pytest.raises(ValueError):
        load_intrinsics(tmp_path / "intrinsics.npy")
