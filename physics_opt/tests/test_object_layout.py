"""Regression tests for object qpos indexing."""

from types import SimpleNamespace

import numpy as np
import torch

from physics_opt.utils.mjwp import _diff_qpos, _get_object_z, _object_qpos_slice


def _freejoint_qpos(nframes: int = 3) -> torch.Tensor:
    robot = np.arange(56, dtype=np.float32) * 0.01
    right = np.array([0.11, 0.22, 0.33, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    left = np.array([0.44, 0.55, 0.66, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return torch.from_numpy(np.stack([np.concatenate([robot, right, left])] * nframes))


def test_object_qpos_slice_bimanual_freejoint():
    qpos = _freejoint_qpos()
    config = SimpleNamespace(nq_obj=14)
    assert torch.allclose(
        _object_qpos_slice(qpos, config, 0),
        torch.tensor([0.11, 0.22, 0.33]).expand(3, 3),
    )
    assert torch.allclose(
        _object_qpos_slice(qpos, config, 1),
        torch.tensor([0.44, 0.55, 0.66]).expand(3, 3),
    )


def test_get_object_z_bimanual_freejoint():
    config = SimpleNamespace(embodiment_type="bimanual", nq_obj=14)
    z = _get_object_z(config, _freejoint_qpos())
    assert torch.allclose(z[:, 0], torch.tensor([0.33] * 3))
    assert torch.allclose(z[:, 1], torch.tensor([0.66] * 3))


def test_diff_qpos_bimanual_freejoint():
    config = SimpleNamespace(embodiment_type="bimanual", nq_obj=14, nv=68, device="cpu")
    reference = _freejoint_qpos()
    simulated = reference.clone()
    simulated[:, 56:59] += torch.tensor([0.01, 0.02, 0.03])
    simulated[:, 63:66] += torch.tensor([-0.05, 0.0, 0.0])

    difference = _diff_qpos(config, simulated, reference)

    assert torch.allclose(difference[0, -12:-9], torch.tensor([0.01, 0.02, 0.03]))
    assert torch.allclose(difference[0, -6:-3], torch.tensor([-0.05, 0.0, 0.0]))
    assert torch.allclose(difference[:, :56], torch.zeros(3, 56))
