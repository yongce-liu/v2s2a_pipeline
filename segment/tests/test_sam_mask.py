"""Tests for multi-prompt SAM mask generation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from segment.anchors import Anchor
from segment.sam_mask import (
    Sam3MaskGenerator,
    SamMaskArgs,
    resolve_text_prompts,
)


class FakeProcessor:
    def __init__(self) -> None:
        self.confidence_threshold = 0.1
        self.set_image_calls = 0
        self.text_prompt_calls: list[str] = []

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = threshold

    def set_image(self, image):
        self.set_image_calls += 1
        return {"backbone_out": object()}

    def set_text_prompt(self, state, prompt: str):
        self.text_prompt_calls.append(prompt)
        mask = np.zeros((1, 6, 8), dtype=np.uint8)
        if prompt == "hand":
            mask[:, 1:3, 1:4] = 1
        else:
            mask[:, 3:5, 4:7] = 1
        state["masks"] = [mask]
        state["scores"] = [0.9]
        return state


def test_segment_prompts_reuses_image_embedding() -> None:
    processor = FakeProcessor()
    generator = Sam3MaskGenerator.__new__(Sam3MaskGenerator)
    generator.device = torch.device("cpu")
    generator.score_threshold = 0.1
    generator.processor = processor
    frame = np.zeros((6, 8, 3), dtype=np.uint8)

    results = generator.segment_prompts(frame, ["hand", "robot arm"])

    assert processor.set_image_calls == 1
    assert processor.text_prompt_calls == ["hand", "robot arm"]
    assert [result.text_prompt for result in results] == ["hand", "robot arm"]
    assert [result.instance_count for result in results] == [1, 1]
    assert results[0].mask[1, 1] == 255
    assert results[1].mask[3, 4] == 255


class FakeAnchorModel:
    def __init__(self) -> None:
        self.predict_calls: list[dict] = []

    def predict_inst(self, inference_state, **kwargs):
        self.predict_calls.append(kwargs)
        masks = np.zeros((3, 6, 8), dtype=bool)
        masks[1, 1:4, 2:5] = True
        return masks, np.asarray([0.1, 0.9, 0.2]), np.empty((3, 256, 256))


class FakeImageProcessor:
    def __init__(self) -> None:
        self.set_image_calls = 0

    def set_image(self, image):
        self.set_image_calls += 1
        return {"backbone_out": {"sam2_backbone_out": object()}}


class FakeAnchorPredictor:
    pass


def test_segment_anchors_reuses_image_embedding_and_selects_best_mask() -> None:
    model = FakeAnchorModel()
    processor = FakeImageProcessor()
    generator = Sam3MaskGenerator.__new__(Sam3MaskGenerator)
    generator.device = torch.device("cpu")
    generator.model = model
    generator.processor = processor
    generator.anchor_predictor = FakeAnchorPredictor()
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    anchor = Anchor(0, (1.0, 2.0, 5.0, 6.0), (3.0, 4.0), 0.9)

    results = generator.segment_anchors(
        frame,
        [("left hand", anchor, "box"), ("right hand", anchor, "box")],
    )

    assert processor.set_image_calls == 1
    assert len(model.predict_calls) == 2
    assert np.array_equal(model.predict_calls[0]["box"], np.asarray([1, 2, 5, 6]))
    assert [result.instance_count for result in results] == [1, 1]
    assert results[0].mask[2, 3] == 255


def test_segment_anchors_supports_positive_points() -> None:
    model = FakeAnchorModel()
    generator = Sam3MaskGenerator.__new__(Sam3MaskGenerator)
    generator.device = torch.device("cpu")
    generator.model = model
    generator.processor = FakeImageProcessor()
    generator.anchor_predictor = FakeAnchorPredictor()
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    anchor = Anchor(0, (1.0, 2.0, 5.0, 6.0), (3.0, 4.0), 0.9)

    generator.segment_anchor(frame, "left hand", anchor, "point")

    call = model.predict_calls[0]
    assert np.array_equal(call["point_coords"], np.asarray([[3, 4]]))
    assert np.array_equal(call["point_labels"], np.asarray([1]))


def test_resolve_text_prompts_supports_legacy_alias() -> None:
    assert resolve_text_prompts(SamMaskArgs(text_prompt="left hand")) == ("left hand",)


def test_resolve_text_prompts_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        resolve_text_prompts(SamMaskArgs(text_prompts=("hand", "hand")))


def test_resolve_text_prompts_rejects_path_separators() -> None:
    with pytest.raises(ValueError, match="path separators"):
        resolve_text_prompts(SamMaskArgs(text_prompts=("hand/left",)))


def test_resolve_text_prompts_rejects_more_than_palette() -> None:
    prompts = tuple(f"object {index}" for index in range(9))
    with pytest.raises(ValueError, match="At most 8"):
        resolve_text_prompts(SamMaskArgs(text_prompts=prompts))
