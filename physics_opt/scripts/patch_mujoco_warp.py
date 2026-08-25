"""Patch the pinned mujoco_warp for mujoco >= 3.5's MjData API rename.

The pinned mujoco_warp commit (5d5a6450) predates mujoco 3.5, where
``MjData.qM`` was renamed to ``MjData.M`` (same packed ``nM`` layout),
``mj_fullM`` changed signature to ``(m, d, dst)``, and the tendon Jacobian
``ten_J`` became dense (the ``ten_J_rownnz``/``ten_J_rowadr``/``ten_J_colind``
sparse metadata was dropped). ``put_data`` in ``mujoco_warp/_src/io.py`` reads
all of these and crashes with ``AttributeError: 'MjData' object has no
attribute 'qM'`` on mujoco >= 3.5.

Run with the package's venv: ``.venv/bin/python scripts/patch_mujoco_warp.py``
(idempotent — skips if the patched markers are already present). Wired as a uv
post-sync hook in ``pyproject.toml`` so ``uv sync`` never silently reverts it.
"""

from __future__ import annotations

import sys
from pathlib import Path

IO_REL = Path("mujoco_warp/_src/io.py")

PATCHES: list[tuple[str, str]] = [
    # 1) sparse/dense mass matrix: qM -> M rename + new mj_fullM signature.
    (
        """  if mujoco.mj_isSparse(mjm):
    qM = np.expand_dims(mjd.qM, axis=0)
    qLD = np.expand_dims(mjd.qLD, axis=0)""",
        """  if mujoco.mj_isSparse(mjm):
    # mujoco >= 3.5 renamed MjData.qM -> M (same packed nM layout)
    _qM_packed = getattr(mjd, "qM", None)
    if _qM_packed is None:
      _qM_packed = mjd.M
    qM = np.expand_dims(_qM_packed, axis=0)
    qLD = np.expand_dims(mjd.qLD, axis=0)""",
    ),
    (
        """    qM = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, qM, mjd.qM)
    if (mjd.qM == 0.0).all() or (mjd.qLD == 0.0).all():""",
        """    qM = np.zeros((mjm.nv, mjm.nv))
    if hasattr(mjd, "qM"):
      mujoco.mj_fullM(mjm, qM, mjd.qM)
    else:
      # mujoco >= 3.5: mj_fullM(m, d, dst); MjData.qM renamed to M
      mujoco.mj_fullM(mjm, mjd, qM)
    _qM_packed = getattr(mjd, "qM", None) if hasattr(mjd, "qM") else mjd.M
    if (_qM_packed == 0.0).all() or (mjd.qLD == 0.0).all():""",
    ),
    # 2) sparse tendon Jacobian: dense (ntendon, nv) since mujoco 3.5.
    (
        """    ten_J = np.zeros((mjm.ntendon, mjm.nv))
    mujoco.mju_sparse2dense(
      ten_J,
      mjd.ten_J.reshape(-1),
      mjd.ten_J_rownnz,
      mjd.ten_J_rowadr,
      mjd.ten_J_colind.reshape(-1),
    )""",
        """    ten_J = np.zeros((mjm.ntendon, mjm.nv))
    if hasattr(mjd, "ten_J_rownnz"):
      mujoco.mju_sparse2dense(
        ten_J,
        mjd.ten_J.reshape(-1),
        mjd.ten_J_rownnz,
        mjd.ten_J_rowadr,
        mjd.ten_J_colind.reshape(-1),
      )
    else:
      # mujoco >= 3.5: tendon Jacobian is dense (ntendon, nv)
      ten_J[:] = mjd.ten_J.reshape((mjm.ntendon, mjm.nv))""",
    ),
    # 3) actuator moment: defensive dense fallback if sparse metadata drops.
    (
        """  actuator_moment = np.zeros((mjm.nu, mjm.nv))
  mujoco.mju_sparse2dense(
    actuator_moment,
    mjd.actuator_moment,
    mjd.moment_rownnz,
    mjd.moment_rowadr,
    mjd.moment_colind,
  )""",
        """  actuator_moment = np.zeros((mjm.nu, mjm.nv))
  if hasattr(mjd, "moment_rownnz") and mjd.moment_rownnz is not None:
    mujoco.mju_sparse2dense(
      actuator_moment,
      mjd.actuator_moment,
      mjd.moment_rownnz,
      mjd.moment_rowadr,
      mjd.moment_colind,
    )
  else:
    actuator_moment[:] = mjd.actuator_moment.reshape((mjm.nu, mjm.nv))""",
    ),
]


def main() -> int:
    try:
        import mujoco_warp
    except ImportError:
        print("mujoco_warp not installed; nothing to patch", file=sys.stderr)
        return 1

    io_path = Path(mujoco_warp.__file__).parent / "_src" / "io.py"
    src = io_path.read_text(encoding="utf-8")
    changed = False
    for old, new in PATCHES:
        if new in src:
            continue  # already patched
        if old not in src:
            print(
                f"warning: expected block not found in {io_path} "
                "(mujoco_warp revision changed?); skipping that patch",
                file=sys.stderr,
            )
            continue
        src = src.replace(old, new)
        changed = True
    if changed:
        io_path.write_text(src, encoding="utf-8")
        print(f"patched {io_path}")
    else:
        print(f"{io_path} already patched (or nothing to do)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
