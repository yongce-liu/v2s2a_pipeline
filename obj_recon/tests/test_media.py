"""Tests for image and mask I/O helpers."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from obj_recon.media import (
    load_mask,
    load_rgb_image,
    load_rgba_image,
    mask_is_empty,
    mask_to_alpha,
)


def _write_test_png(path: Path, data: np.ndarray):
    Image.fromarray(data).save(str(path))


def test_load_rgb_image():
    """load_rgb_image reads a 3-channel image as RGB."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.png"
        bgr = np.zeros((32, 32, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # B channel
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(path), bgr)
        result = load_rgb_image(path)
        assert result.shape == (32, 32, 3)
        assert np.array_equal(result, rgb)


def test_load_rgb_image_missing():
    with pytest.raises(FileNotFoundError):
        load_rgb_image(Path("/nonexistent/image.png"))


def test_load_mask_bool():
    """load_mask returns a boolean array."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        data = np.zeros((16, 16), dtype=np.uint8)
        data[4:8, 4:8] = 255
        _write_test_png(path, data)
        mask = load_mask(path)
        assert mask.shape == (16, 16)
        assert mask.dtype == bool
        assert mask[5, 5]


def test_load_mask_missing():
    with pytest.raises(FileNotFoundError):
        load_mask(Path("/nonexistent/mask.png"))


def test_mask_is_empty():
    empty = np.zeros((10, 10), dtype=bool)
    assert mask_is_empty(empty)
    not_empty = np.zeros((10, 10), dtype=bool)
    not_empty[2, 2] = True
    assert not mask_is_empty(not_empty)


def test_mask_to_alpha():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    alpha = mask_to_alpha(mask)
    assert alpha.shape == (8, 8)
    assert alpha.dtype == np.uint8
    assert alpha[3, 3] == 255
    assert alpha[0, 0] == 0


def test_load_rgba_image_rgb_fallback():
    """When source has no alpha, an opaque channel is appended."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rgb.png"
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        Image.fromarray(rgb).save(str(path))
        result = load_rgba_image(path)
        assert result.shape == (16, 16, 4)
        assert result[0, 0, 3] == 255


# Import cv2 locally for the test above
import cv2
import pytest
