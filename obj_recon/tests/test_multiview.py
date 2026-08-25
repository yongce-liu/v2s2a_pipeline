"""Unit tests for the MV-SAM3D keyframe selection helpers (no model / CUDA)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sam3d_objects.model.backbone.generator.flow_matching.solver import Euler
from sam3d_objects.pipeline.multi_view_utils import inject_generator_multi_view

from obj_recon.multiview import (
    _evenly_spaced,
    _normalize_intrinsics,
    _view_pose_record,
    select_keyframes,
)


class TestEvenlySpaced:
    def test_fewer_candidates_than_n_returns_all(self):
        assert _evenly_spaced([0, 44], 4) == [0, 44]

    def test_exact_count(self):
        assert _evenly_spaced([0, 22, 44, 88], 4) == [0, 22, 44, 88]

    def test_spans_endpoints(self):
        out = _evenly_spaced(list(range(0, 89)), 4)
        assert out[0] == 0 and out[-1] == 88
        assert len(out) == 4

    def test_downsamples_to_n(self):
        out = _evenly_spaced(list(range(0, 89)), 3)
        assert len(out) == 3


def test_normalized_intrinsics_are_resize_invariant():
    intrinsics = np.array([[900.0, 0.0, 960.0], [0.0, 880.0, 540.0], [0.0, 0.0, 1.0]])
    full = _normalize_intrinsics(intrinsics, 1920, 1080)
    resized_intrinsics = intrinsics.copy()
    resized_intrinsics[0, :] *= 0.5
    resized_intrinsics[1, :] *= 0.5
    half = _normalize_intrinsics(resized_intrinsics, 960, 540)
    np.testing.assert_allclose(full, half)


def test_view_pose_record_converts_axes_and_canonical_basis():
    record = _view_pose_record(
        view=0,
        frame_index=44,
        rotation=[1.0, 0.0, 0.0, 0.0],
        translation=[1.0, 2.0, 3.0],
        scale=[0.2, 0.2, 0.2],
        reference=True,
        fit_iou=0.8,
    )
    transform = np.asarray(record["object_to_camera_opencv"])
    expected_rotation = np.diag([-1.0, -1.0, 1.0]) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    np.testing.assert_allclose(transform[:3, :3], expected_rotation)
    np.testing.assert_allclose(transform[:3, 3], [-1.0, -2.0, 3.0])


class _FakeGenerator:
    inference_steps = 1
    _solver = Euler()

    @staticmethod
    def _prepare_t(steps=None):
        return torch.tensor([0.0, 1.0])

    @staticmethod
    def _generate_dynamics(state, _t, condition):
        return {
            "shape": torch.full_like(state["shape"], float(condition)),
            "translation": torch.full_like(state["translation"], float(condition)),
        }


def test_multiview_tracks_independent_pose_states():
    generator = _FakeGenerator()
    state = {
        "shape": torch.zeros(1, 1),
        "translation": torch.zeros(1, 3),
    }
    conditions = torch.tensor([1.0, 3.0])
    with inject_generator_multi_view(generator, 2, 1) as poses:
        velocity = generator._generate_dynamics(state, 0.0, conditions)

    torch.testing.assert_close(velocity["shape"], torch.tensor([[2.0]]))
    torch.testing.assert_close(velocity["translation"], torch.ones(1, 3))
    torch.testing.assert_close(
        poses["translation"], torch.tensor([[[1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0]]])
    )


class TestSelectKeyFrames:
    USABLE = list(range(0, 89))  # 0..88

    def test_even_strategy(self):
        k = select_keyframes(self.USABLE, strategy="even", num_views=4, max_views_cap=8)
        assert k[0] == 0 and len(k) == 4 and k[-1] == 88

    def test_even_cap(self):
        k = select_keyframes(
            self.USABLE, strategy="even", num_views=12, max_views_cap=6
        )
        assert len(k) <= 6

    def test_manual_preserves_order(self):
        k = select_keyframes(
            self.USABLE,
            strategy="manual",
            num_views=4,
            max_views_cap=8,
            manual=[10, 44, 0],
        )
        assert k == [10, 44, 0]  # user order — first is view-0 reference

    def test_manual_filters_unusable(self):
        k = select_keyframes(
            self.USABLE,
            strategy="manual",
            num_views=4,
            max_views_cap=8,
            manual=[0, 9999, 44],
        )
        assert set(k) == {0, 44} and 9999 not in k

    def test_manual_requires_list(self):
        with pytest.raises(ValueError, match="manual"):
            select_keyframes(
                self.USABLE,
                strategy="manual",
                num_views=4,
                max_views_cap=8,
                manual=None,
            )

    def test_manual_too_few_usable_raises(self):
        with pytest.raises(ValueError, match=">=2 usable"):
            select_keyframes(
                self.USABLE, strategy="manual", num_views=4, max_views_cap=8, manual=[0]
            )

    def test_unknown_strategy_falls_back_to_even(self):
        k = select_keyframes(
            self.USABLE, strategy="nonsense", num_views=4, max_views_cap=8
        )
        assert k[0] == 0 and len(k) == 4
