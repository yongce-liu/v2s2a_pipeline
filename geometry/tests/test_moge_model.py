"""Tests for the MogeArgs validation and checkpoint resolution (no GPU needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from geometry.moge_model import MogeArgs, MogeModel


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    args = MogeArgs(checkpoint=tmp_path / "missing.pt", allow_hf_download=False)
    with pytest.raises(FileNotFoundError):
        MogeModel(args)


def test_default_checkpoint_points_at_repo_weights() -> None:
    args = MogeArgs()
    assert args.version == "v3"
    # The default path is repo-relative; it may or may not exist locally, but
    # it must point into the pipeline's weights directory.
    assert "weights" in str(args.checkpoint)
    assert args.checkpoint.name == "model.pt"
    assert args.checkpoint.parent.name == "moge-3-vitg"
