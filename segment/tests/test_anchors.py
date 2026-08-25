"""Tests for generic geometric anchor manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from segment.anchors import load_anchor_manifest


def _write_anchor_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "coordinate_space": "pixel",
                "box_format": "xyxy",
                "targets": [
                    {"prompt_id": "left hand", "anchor_type": "box"},
                    {"prompt_id": "object tip", "anchor_type": "point"},
                ],
                "entries": [
                    {
                        "index": 4,
                        "anchors": {
                            "left hand": {
                                "box_xyxy": [1, 2, 5, 8],
                                "point_xy": [3, 5],
                                "confidence": 0.75,
                            },
                            "object tip": {
                                "point_xy": [20, 30],
                                "confidence": 0.9,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_anchor_manifest_preserves_target_order_and_types(tmp_path: Path) -> None:
    path = tmp_path / "anchors.json"
    _write_anchor_manifest(path)

    manifest = load_anchor_manifest(path)

    assert [target.prompt_id for target in manifest.targets] == [
        "left hand",
        "object tip",
    ]
    assert [target.anchor_type for target in manifest.targets] == ["box", "point"]
    assert manifest.anchors["left hand"][4].box_xyxy == (1.0, 2.0, 5.0, 8.0)
    assert manifest.anchors["object tip"][4].point_xy == (20.0, 30.0)


def test_load_anchor_manifest_rejects_missing_selected_geometry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "anchors.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_space": "pixel",
                "box_format": "xyxy",
                "targets": [{"prompt_id": "hand", "anchor_type": "point"}],
                "entries": [
                    {
                        "index": 0,
                        "anchors": {"hand": {"box_xyxy": [1, 2, 3, 4]}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing point anchor"):
        load_anchor_manifest(path)
