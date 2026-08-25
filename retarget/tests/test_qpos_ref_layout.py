"""Smoke test for the (H, sites, 7) qpos_ref reference layout used by mink IK."""

from __future__ import annotations

import numpy as np


def test_qpos_ref_concat_layout():
    h, n_finger = 4, 5
    wrist_r = np.zeros((h, 7))
    finger_r = np.zeros((h, n_finger, 7))
    wrist_l = np.ones((h, 7))
    finger_l = np.ones((h, n_finger, 7))
    obj_r = np.full((h, 7), 2.0)
    obj_l = np.full((h, 7), 3.0)
    qpos_ref = np.concatenate(
        [
            wrist_r[:, None],
            finger_r,
            wrist_l[:, None],
            finger_l,
            obj_r[:, None],
            obj_l[:, None],
        ],
        axis=1,
    )
    assert qpos_ref.shape == (h, 14, 7)
    # layout: 0 right_palm, 1-5 right fingers, 6 left_palm, 7-11 left fingers,
    # 12 right_object, 13 left_object (mirrors solve_ik.py ref_idx map)
    np.testing.assert_array_equal(qpos_ref[:, 6], 1.0)  # left palm
    np.testing.assert_array_equal(qpos_ref[:, 12], 2.0)  # right object
    np.testing.assert_array_equal(qpos_ref[:, 13], 3.0)  # left object
