"""Tests for the video-mode (frame-by-frame) segmentation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from segment.anchors import Anchor
from segment.media import load_mask
from segment.sam_mask import PromptMaskResult, SamMaskArgs
from segment.workflow import (
    SegmentVideoArgs,
    SegmentVideoOutputs,
    run_video_segment,
)
from tests.test_frames import build_process_layout


class FakeGenerator:
    """Duck-typed stand-in for Sam3MaskGenerator (no SAM3 model needed)."""

    def __init__(
        self,
        masks: np.ndarray | tuple[np.ndarray, ...],
        instance_counts: int | tuple[int, ...] = 1,
    ) -> None:
        self._masks = (masks,) if isinstance(masks, np.ndarray) else masks
        self._instance_counts = (
            (instance_counts,) if isinstance(instance_counts, int) else instance_counts
        )
        self.segment_calls: list[tuple[tuple[int, int], tuple[str, ...]]] = []
        self.anchor_calls: list[
            tuple[tuple[int, int], tuple[tuple[str, str], ...]]
        ] = []

    def segment_prompts(
        self, frame_rgb: np.ndarray, text_prompts: tuple[str, ...]
    ) -> list[PromptMaskResult]:
        self.segment_calls.append((frame_rgb.shape[:2], tuple(text_prompts)))
        masks = self._masks
        counts = self._instance_counts
        if len(masks) == 1:
            masks = masks * len(text_prompts)
        if len(counts) == 1:
            counts = counts * len(text_prompts)
        assert len(masks) == len(text_prompts)
        assert len(counts) == len(text_prompts)
        return [
            PromptMaskResult(prompt, mask.copy(), count)
            for prompt, mask, count in zip(text_prompts, masks, counts)
        ]

    def segment_anchors(
        self,
        frame_rgb: np.ndarray,
        anchors: list[tuple[str, Anchor, str]],
    ) -> list[PromptMaskResult]:
        self.anchor_calls.append(
            (
                frame_rgb.shape[:2],
                tuple((name, anchor_type) for name, _, anchor_type in anchors),
            )
        )
        masks = self._masks
        if len(masks) == 1:
            masks = masks * len(anchors)
        return [
            PromptMaskResult(name, mask.copy(), 1)
            for (name, _, _), mask in zip(anchors, masks)
        ]


def _make_fake_generator(frame_shape=(8, 10), instance_count=1):
    mask = np.zeros(frame_shape, dtype=np.uint8)
    mask[2:6, 1:5] = 255
    return FakeGenerator(mask, instance_count)


def _build_anchors_json(tmp_path: Path, frame_count: int) -> Path:
    path = tmp_path / "anchors.json"
    path.write_text(
        json.dumps(
            {
                "coordinate_space": "pixel",
                "box_format": "xyxy",
                "targets": [{"prompt_id": "left hand", "anchor_type": "box"}],
                "entries": [
                    {
                        "index": index,
                        "anchors": {
                            "left hand": {
                                "box_xyxy": [1, 2, 4, 7],
                                "point_xy": [2.5, 4.5],
                                "confidence": 0.8,
                            }
                        },
                    }
                    for index in range(frame_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_run_video_segment_with_vis(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=3)
    fake = _make_fake_generator()

    outputs = run_video_segment(
        SegmentVideoArgs(
            frames_json=frames_json,
            output_root=tmp_path,
            vis=True,
        ),
        generator=fake,
    )

    assert isinstance(outputs, SegmentVideoOutputs)
    assert outputs.stage_dir == tmp_path / "0" / "segment"
    assert outputs.masks_dir.exists()
    assert outputs.masks_vis_dir is not None and outputs.masks_vis_dir.exists()

    masks = sorted(p.name for p in outputs.masks_dir.glob("*.png"))
    vis = sorted(p.name for p in outputs.masks_vis_dir.glob("*.jpg"))
    assert masks == ["000000.png", "000001.png", "000002.png"]
    assert vis == ["000000.jpg", "000001.jpg", "000002.jpg"]
    assert sorted(
        p.name for p in (outputs.masks_dir / "human hand and arm").glob("*.png")
    ) == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]

    manifest = json.loads(outputs.masks_json_path.read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 3
    assert manifest["processed_count"] == 3
    assert manifest["vis_enabled"] is True
    assert manifest["masks_vis_dir"] == str(outputs.masks_vis_dir)
    assert len(manifest["entries"]) == 3

    entry = manifest["entries"][0]
    assert entry["index"] == 0
    assert entry["frame_filename"] == "000000.png"
    assert entry["mask_filename"] == "000000.png"
    assert entry["vis_filename"] == "000000.jpg"
    assert entry["has_mask"] is True
    assert entry["instance_count"] == 1
    assert entry["area"] == 4 * 4
    assert entry["bbox"] == {"min_row": 2, "min_col": 1, "max_row": 5, "max_col": 4}
    assert entry["prompt_masks"] == [
        {
            "prompt_id": "human hand and arm",
            "text_prompt": "human hand and arm",
            "input_type": "text",
            "anchor": None,
            "mask_filename": "human hand and arm/000000.png",
            "has_mask": True,
            "instance_count": 1,
            "area": 16,
            "bbox": {
                "min_row": 2,
                "min_col": 1,
                "max_row": 5,
                "max_col": 4,
            },
        }
    ]

    config = json.loads(outputs.config_json_path.read_text(encoding="utf-8"))
    assert config["package"]["name"] == "segment"
    assert config["source"]["frame_count"] == 3
    assert config["segment"]["vis"] is True
    assert config["segment"]["text_prompts"] == ["human hand and arm"]
    assert config["segment"]["prompt_colors_rgb"] == [[0, 0, 255]]

    # The generator was reused for every frame (model loaded once).
    assert len(fake.segment_calls) == 3


def test_run_video_segment_multiple_prompts(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=1)
    first = np.zeros((8, 10), dtype=np.uint8)
    first[1:3, 1:4] = 255
    second = np.zeros((8, 10), dtype=np.uint8)
    second[4:7, 6:9] = 255
    fake = FakeGenerator((first, second), (2, 1))

    outputs = run_video_segment(
        SegmentVideoArgs(
            frames_json=frames_json,
            output_root=tmp_path,
            sam_mask=SamMaskArgs(text_prompts=("human hand", "robot gripper")),
        ),
        generator=fake,
    )

    assert fake.segment_calls == [((8, 10), ("human hand", "robot gripper"))]
    assert np.array_equal(
        load_mask(outputs.masks_dir / "human hand" / "000000.png"), first
    )
    assert np.array_equal(
        load_mask(outputs.masks_dir / "robot gripper" / "000000.png"), second
    )
    assert np.array_equal(
        load_mask(outputs.masks_dir / "000000.png"), np.maximum(first, second)
    )

    manifest = json.loads(outputs.masks_json_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.2"
    assert manifest["prompts"] == [
        {
            "prompt_id": "human hand",
            "text_prompt": "human hand",
            "input_type": "text",
            "anchor_source": None,
            "color_rgb": [0, 0, 255],
            "masks_dir": str(outputs.masks_dir / "human hand"),
        },
        {
            "prompt_id": "robot gripper",
            "text_prompt": "robot gripper",
            "input_type": "text",
            "anchor_source": None,
            "color_rgb": [235, 104, 52],
            "masks_dir": str(outputs.masks_dir / "robot gripper"),
        },
    ]
    prompt_masks = manifest["entries"][0]["prompt_masks"]
    assert [item["text_prompt"] for item in prompt_masks] == [
        "human hand",
        "robot gripper",
    ]
    assert [item["instance_count"] for item in prompt_masks] == [2, 1]
    assert manifest["entries"][0]["instance_count"] == 3
    assert manifest["entries"][0]["area"] == 15


def test_run_video_segment_with_anchor_json_and_text(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=1)
    anchors_json = _build_anchors_json(tmp_path, frame_count=1)
    hand_mask = np.zeros((8, 10), dtype=np.uint8)
    hand_mask[2:7, 1:4] = 255
    fake = FakeGenerator(hand_mask)

    outputs = run_video_segment(
        SegmentVideoArgs(
            frames_json=frames_json,
            output_root=tmp_path,
            anchors_json=anchors_json,
            sam_mask=SamMaskArgs(text_prompts=("yellow spoon",)),
        ),
        generator=fake,
    )

    assert fake.segment_calls == [((8, 10), ("yellow spoon",))]
    assert fake.anchor_calls == [((8, 10), (("left hand", "box"),))]
    manifest = json.loads(outputs.masks_json_path.read_text(encoding="utf-8"))
    assert manifest["anchor_source"] == str(anchors_json.resolve())
    assert manifest["prompts"][0]["input_type"] == "anchor_box"
    assert manifest["prompts"][0]["anchor_source"] == "json"
    assert manifest["prompts"][0]["text_prompt"] is None
    assert manifest["prompts"][1]["input_type"] == "text"

    by_id = {item["prompt_id"]: item for item in manifest["entries"][0]["prompt_masks"]}
    assert by_id["left hand"]["text_prompt"] is None
    assert by_id["left hand"]["input_type"] == "anchor_box"
    assert by_id["left hand"]["anchor"] == {
        "point_xy": [2.5, 4.5],
        "box_xyxy": [1.0, 2.0, 4.0, 7.0],
        "confidence": pytest.approx(0.8),
    }
    assert by_id["yellow spoon"]["input_type"] == "text"
    assert by_id["yellow spoon"]["anchor"] is None


def test_run_video_segment_anchor_json_suppresses_default_text_prompt(
    tmp_path: Path,
) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=1)
    anchors_json = _build_anchors_json(tmp_path, frame_count=1)
    fake = _make_fake_generator()

    outputs = run_video_segment(
        SegmentVideoArgs(
            frames_json=frames_json,
            output_root=tmp_path,
            anchors_json=anchors_json,
        ),
        generator=fake,
    )

    assert fake.segment_calls == []
    assert fake.anchor_calls == [((8, 10), (("left hand", "box"),))]
    manifest = json.loads(outputs.masks_json_path.read_text(encoding="utf-8"))
    assert [item["prompt_id"] for item in manifest["prompts"]] == ["left hand"]


def test_run_video_segment_without_vis(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=2)
    outputs = run_video_segment(
        SegmentVideoArgs(frames_json=frames_json, output_root=tmp_path, vis=False),
        generator=_make_fake_generator(),
    )

    assert outputs.masks_vis_dir is None
    assert not (outputs.stage_dir / "masks_vis").exists()
    manifest = json.loads(outputs.masks_json_path.read_text(encoding="utf-8"))
    assert manifest["vis_enabled"] is False
    assert manifest["masks_vis_dir"] is None
    assert all(e["vis_filename"] is None for e in manifest["entries"])


def test_run_video_segment_max_frames(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=5)
    fake = _make_fake_generator()
    outputs = run_video_segment(
        SegmentVideoArgs(frames_json=frames_json, output_root=tmp_path, max_frames=2),
        generator=fake,
    )

    assert len(outputs.entries) == 2
    manifest = json.loads(outputs.masks_json_path.read_text(encoding="utf-8"))
    assert manifest["processed_count"] == 2
    assert [e["index"] for e in manifest["entries"]] == [0, 1]


def test_run_video_segment_empty_mask(tmp_path: Path) -> None:
    frames_json, _clip_root = build_process_layout(tmp_path, frame_count=1)
    fake = FakeGenerator(np.zeros((8, 10), dtype=np.uint8), instance_counts=0)
    outputs = run_video_segment(
        SegmentVideoArgs(frames_json=frames_json, output_root=tmp_path),
        generator=fake,
    )

    entry = outputs.entries[0]
    assert not entry.has_mask
    assert entry.instance_count == 0
    assert entry.area == 0
    assert entry.bbox is None


def test_run_video_segment_missing_frames_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_video_segment(
            SegmentVideoArgs(frames_json=tmp_path / "nope.json", output_root=tmp_path),
            generator=_make_fake_generator(),
        )


def test_run_video_segment_negative_max_frames(tmp_path: Path) -> None:
    frames_json, _ = build_process_layout(tmp_path)
    with pytest.raises(ValueError):
        run_video_segment(
            SegmentVideoArgs(
                frames_json=frames_json, output_root=tmp_path, max_frames=-1
            ),
            generator=_make_fake_generator(),
        )
