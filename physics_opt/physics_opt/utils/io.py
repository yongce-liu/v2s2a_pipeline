"""Define functions to load and save the data."""

from __future__ import annotations

import contextlib
import errno
import glob
import os
import time
from typing import TYPE_CHECKING


import loguru
import numpy as np
import torch
from filelock import FileLock

from physics_opt.utils.interp import align_to_sim_dt


@contextlib.contextmanager
def nfs_safe_lock(lock_path: str, timeout: float = 600.0, max_retries: int = 5):
    """FileLock that retries on NFS ESTALE during acquisition.

    fcntl.flock occasionally returns ESTALE (Errno 116) on NFS when the lock
    file's inode is replaced under us by a sibling process. Retrying with a
    fresh FileLock instance recovers; the underlying flock is otherwise fine.
    """
    delay = 0.5
    for attempt in range(max_retries):
        lock = FileLock(lock_path, timeout=timeout)
        try:
            lock.acquire()
        except OSError as e:
            if e.errno != errno.ESTALE or attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        try:
            yield
        finally:
            lock.release()
        return


if TYPE_CHECKING:
    from physics_opt.config import Config


def load_data(
    config: Config,
    data_path: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_data = np.load(data_path)
    qpos_ref = raw_data["qpos"]
    qvel_ref = raw_data["qvel"]
    if config.contact_rew_scale > 0.0:
        if "contact" not in raw_data:
            raise ValueError("contact data not found while contact_rew_scale > 0.0")
        if "contact_pos" not in raw_data:
            raise ValueError("contact_pos data not found while contact_rew_scale > 0.0")
    try:
        contact = raw_data["contact"]
    except KeyError:
        contact = np.zeros((qpos_ref.shape[0], 10))
        loguru.logger.warning("contact data not found")
    try:
        contact_pos = raw_data["contact_pos"]
    except KeyError:
        contact_pos = np.zeros((qpos_ref.shape[0], 10, 3))
        loguru.logger.warning("contact_pos data not found")
    if "ctrl" in raw_data:
        ctrl_ref = raw_data["ctrl"]
    else:
        loguru.logger.warning(
            "ctrl data not found, using 'qpos' as a initial guess for control."
        )
        if config.embodiment_type in ["bimanual", "right", "left"]:
            ctrl_ref = qpos_ref[:, : -config.nq_obj]
        else:
            raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    qpos_ref_torch = torch.from_numpy(qpos_ref).to(config.device).to(torch.float32)
    qvel_ref_torch = torch.from_numpy(qvel_ref).to(config.device).to(torch.float32)
    ctrl_ref_torch = torch.from_numpy(ctrl_ref).to(config.device).to(torch.float32)
    contact_ref_torch = torch.from_numpy(contact).to(config.device).to(torch.float32)
    contact_pos_ref_torch = (
        torch.from_numpy(contact_pos).to(config.device).to(torch.float32)
    )
    # align reference to sim_dt — see retargeting/utils/interp.py:align_to_sim_dt
    qpos_ref_interp = align_to_sim_dt(qpos_ref_torch, config.ref_dt, config.sim_dt)
    qvel_ref_interp = align_to_sim_dt(qvel_ref_torch, config.ref_dt, config.sim_dt)
    ctrl_ref_interp = align_to_sim_dt(ctrl_ref_torch, config.ref_dt, config.sim_dt)
    contact_ref_interp = align_to_sim_dt(
        contact_ref_torch, config.ref_dt, config.sim_dt
    )
    H, Nc, D = contact_pos_ref_torch.shape
    contact_pos_ref_flat = contact_pos_ref_torch.view(H, Nc * D)
    contact_pos_ref_interp = align_to_sim_dt(
        contact_pos_ref_flat, config.ref_dt, config.sim_dt
    ).view(-1, Nc, D)
    # prepend warmup frames: reference is frame 0 (closed grasp) throughout;
    # the initial sim state will be set to open hand in optimize_physics.py
    if config.warmup_steps > 0:
        n = config.warmup_steps
        qpos_ref_interp = torch.cat(
            [qpos_ref_interp[:1].expand(n, -1), qpos_ref_interp], dim=0
        )
        qvel_ref_interp = torch.cat(
            [
                torch.zeros(
                    n,
                    qvel_ref_interp.shape[1],
                    device=qvel_ref_interp.device,
                    dtype=qvel_ref_interp.dtype,
                ),
                qvel_ref_interp,
            ],
            dim=0,
        )
        ctrl_ref_interp = torch.cat(
            [ctrl_ref_interp[:1].expand(n, -1), ctrl_ref_interp], dim=0
        )
        contact_ref_interp = torch.cat(
            [contact_ref_interp[:1].expand(n, -1), contact_ref_interp], dim=0
        )
        contact_pos_ref_interp = torch.cat(
            [contact_pos_ref_interp[:1].expand(n, -1, -1), contact_pos_ref_interp],
            dim=0,
        )
        loguru.logger.info("Prepended {} warmup frames to reference trajectory.", n)
    # repeat the last frame with extra config.horizon_steps
    for _ in range(config.horizon_steps + config.ctrl_steps):
        qpos_ref_interp = torch.cat([qpos_ref_interp, qpos_ref_interp[-1:]], dim=0)
        qvel_ref_interp = torch.cat([qvel_ref_interp, qvel_ref_interp[-1:]], dim=0)
        ctrl_ref_interp = torch.cat([ctrl_ref_interp, ctrl_ref_interp[-1:]], dim=0)
        contact_ref_interp = torch.cat(
            [contact_ref_interp, contact_ref_interp[-1:]], dim=0
        )
        contact_pos_ref_interp = torch.cat(
            [contact_pos_ref_interp, contact_pos_ref_interp[-1:]], dim=0
        )
    return (
        qpos_ref_interp,
        qvel_ref_interp,
        ctrl_ref_interp,
        contact_ref_interp,
        contact_pos_ref_interp,
    )


def warmup_ref_interp(
    init_qpos: np.ndarray,
    target_qpos: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Linearly interpolate open-hand init to closed-grasp frame 0 over n_steps."""
    if n_steps <= 0:
        return init_qpos[np.newaxis, :0]
    alpha = np.linspace(0, 1, n_steps + 1, dtype=np.float32)[:-1, np.newaxis]
    return init_qpos[np.newaxis] * (1 - alpha) + target_qpos[np.newaxis] * alpha


def get_processed_data_dir(
    output_root_dir: str,
    dataset_name: str,
    robot_type: str,
    embodiment_type: str,
    task: str,
    data_id: int,
) -> str:
    return f"{output_root_dir}/{robot_type}/{embodiment_type}/{task}/{data_id}"


def get_mesh_dir(output_root_dir: str, dataset_name: str, object_name: str) -> str:
    return f"{output_root_dir}/assets/objects/{object_name}"


def resolve_auto_embodiment(dataset_name: str, output_root_dir: str, task: str) -> str:
    """Resolve "auto" embodiment_type from the processed data layout."""
    if not dataset_name.startswith("do_as_i_do"):
        return "bimanual"

    pattern = os.path.join(os.path.abspath(output_root_dir), "mano", "*", task)
    embodiments = sorted(
        {
            os.path.basename(os.path.dirname(d))
            for d in glob.glob(pattern)
            if os.path.isdir(d)
        }
    )
    if not embodiments:
        raise FileNotFoundError(f"No processed output for task '{task}'; run stage 1")
    if len(embodiments) > 1:
        raise RuntimeError(f"Task '{task}' found multiple embodiments {embodiments}")
    return embodiments[0]
