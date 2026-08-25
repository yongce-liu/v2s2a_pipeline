"""Utils for mujoco."""

from contextlib import contextmanager

import mujoco
import mujoco.viewer
import numpy as np


def get_viewer(show_viewer: bool, model: mujoco.MjModel, data: mujoco.MjData):
    if show_viewer:

        def run_viewer():
            return mujoco.viewer.launch_passive(model, data)
    else:
        cam = mujoco.MjvCamera()
        cam.type = 2
        cam.fixedcamid = 0

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {"is_running": lambda: True, "sync": lambda: None, "cam": 0},
            )

    return run_viewer


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo wxyz quaternion to a 3x3 rotation matrix."""
    q = q_wxyz / (np.linalg.norm(q_wxyz) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def get_palm_geometry_from_model(
    model: mujoco.MjModel, side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Palm (offset, normal, quat_wxyz) from a compiled MjModel, or None.

    Normal is the site frame's x-axis in the parent body frame.
    """
    site_name = f"{side}_palm"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        return None
    offset = model.site_pos[site_id].copy()
    q_wxyz = model.site_quat[site_id].copy()
    R = quat_wxyz_to_rotmat(q_wxyz)
    normal = R[:, 0].copy()
    return offset, normal, q_wxyz
