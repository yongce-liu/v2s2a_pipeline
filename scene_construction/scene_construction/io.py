"""Shared do-as-i-do ``outputs/`` layout helpers and the NFS-safe lock."""

from __future__ import annotations

import contextlib
import errno
import glob
import os
import time

from filelock import FileLock


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
