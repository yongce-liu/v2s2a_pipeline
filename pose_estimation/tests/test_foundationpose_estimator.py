"""CPU-only tests for the FoundationPose wrapper lifecycle."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pose_estimation.foundationpose_estimator import FoundationPoseEstimator


class _FakeUpstreamEstimator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.pose_last = None
        self.best_id = None
        self.poses = None
        self.scores = None
        self.retained_refs: list[
            tuple[weakref.ReferenceType, weakref.ReferenceType]
        ] = []

    def register(self, **_kwargs) -> np.ndarray:
        poses = torch.zeros((252, 4, 4))
        scores = torch.zeros(252)
        poses[:, 3, 3] = 1.0
        self.pose_last = poses[0]
        self.best_id = torch.tensor(0)
        self.poses = poses
        self.scores = scores
        self.retained_refs.append((weakref.ref(poses), weakref.ref(scores)))
        if self.fail:
            raise RuntimeError("registration failed")
        return np.eye(4)


def _wrapper(upstream: _FakeUpstreamEstimator) -> FoundationPoseEstimator:
    wrapper = object.__new__(FoundationPoseEstimator)
    wrapper.estimator = upstream
    wrapper.args = SimpleNamespace(est_refine_iter=5)
    wrapper._registered = False
    return wrapper


def _register(wrapper: FoundationPoseEstimator) -> np.ndarray:
    return wrapper.register(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        np.eye(3),
    )


def test_register_releases_upstream_hypotheses() -> None:
    upstream = _FakeUpstreamEstimator()
    wrapper = _wrapper(upstream)

    result = _register(wrapper)
    gc.collect()

    poses_ref, scores_ref = upstream.retained_refs[0]
    assert np.array_equal(result, np.eye(4))
    assert wrapper._registered
    assert upstream.poses is None
    assert upstream.scores is None
    assert upstream.pose_last._base is None
    assert poses_ref() is None
    assert scores_ref() is None


def test_repeated_register_does_not_retain_previous_hypotheses() -> None:
    upstream = _FakeUpstreamEstimator()
    wrapper = _wrapper(upstream)

    _register(wrapper)
    _register(wrapper)
    gc.collect()

    assert all(ref() is None for pair in upstream.retained_refs for ref in pair)
    assert upstream.pose_last._base is None


def test_set_pose_composes_mesh_center_in_object_frame(monkeypatch) -> None:
    upstream = _FakeUpstreamEstimator()
    upstream.model_center = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    wrapper = _wrapper(upstream)
    monkeypatch.setattr(
        torch,
        "as_tensor",
        lambda value, device=None: np.asarray(value),
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose[:3, 3] = [0.0, 0.0, 2.0]

    wrapper.set_pose(pose)

    np.testing.assert_allclose(upstream.pose_last[:3, 3], [0.0, 1.0, 2.0])
    assert wrapper._registered


def test_register_releases_hypotheses_after_failure() -> None:
    upstream = _FakeUpstreamEstimator(fail=True)
    wrapper = _wrapper(upstream)

    with pytest.raises(RuntimeError, match="registration failed"):
        _register(wrapper)
    gc.collect()

    poses_ref, scores_ref = upstream.retained_refs[0]
    assert not wrapper._registered
    assert upstream.poses is None
    assert upstream.scores is None
    assert upstream.pose_last._base is None
    assert poses_ref() is None
    assert scores_ref() is None
