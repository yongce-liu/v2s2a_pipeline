"""Smoke tests for scene_construction.in_hand proximity masks."""

from __future__ import annotations

import numpy as np

from scene_construction.in_hand import (
    compute_near_floor_mask,
    erode_mask,
    hand_object_min_dist,
)


def _box_verts(half: float = 0.05) -> np.ndarray:
    s = np.array([-half, half])
    return np.array([[x, y, z] for x in s for y in s for z in s])


def test_hand_object_min_dist_at_origin():
    # hand point on the box axis: distance to the nearest of the 4 top-face
    # corners (each at x^2 + y^2 = (0.05*sqrt(2))^2 out from the axis)
    corner_r = 0.05 * np.sqrt(2)
    above = 0.01
    hand = np.array([[0.0, 0.0, 0.05 + above]])
    qpos = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # identity pose
    dist, *_ = hand_object_min_dist(hand, qpos, _box_verts())
    assert abs(dist - float(np.hypot(corner_r, above))) < 1e-9


def test_near_floor_mask_tracks_lowest_vertex():
    qpos = np.tile(
        np.array([0.0, 0.0, 0.06, 1.0, 0.0, 0.0, 0.0]), (3, 1)
    )  # bottom face at z=0.01
    mask = compute_near_floor_mask(qpos, _box_verts(), distance_thresh=0.02)
    assert mask.all()
    far = qpos.copy()
    far[:, 2] = 0.5
    assert not compute_near_floor_mask(far, _box_verts(), distance_thresh=0.02).any()


def test_erode_mask_symmetric_window():
    mask = np.array([0, 1, 1, 1, 1, 1, 0], dtype=bool)
    out = erode_mask(mask, steps=3)
    # window [t-2, t+2] must be fully True — only the middle frame qualifies
    assert out.tolist() == [False, False, False, True, False, False, False]
