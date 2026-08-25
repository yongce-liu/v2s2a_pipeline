"""Fuse/filter a pose_estimation track into a damage-limited trajectory.

Reads an existing ``poses.json`` (no GPU, no renderer), detects jump spans,
bridges them between healthy pivots (flagged ``fused-bridge`` / ``fused-hold``
/ ``fused-keep`` methods in the fused manifest), and writes:

.. code-block:: text

    <stage_dir>/
    ├── poses_fused.json        # same schema as poses.json + fused verdicts
    ├── poses_fused/            # fused 4x4 pose matrices
    │   ├── 000000.txt
    │   └── ...
    └── track_metrics.json      # before/after innovation + stability metrics

Usage:

    uv run python -m pose_estimation.temporal_cli \
        --poses-json outputs/yellow_spoon/pose_estimation/poses.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

from pose_estimation.temporal import (
    JumpGateConfig,
    fuse_track,
    gate_steps,
    load_poses,
    pose_stability_metrics,
    track_metrics,
)


@dataclass
class TemporalFuseArgs:
    poses_json: Path
    """Path to the stage's ``poses.json`` manifest."""

    output_dir: Path | None = None
    """Where ``poses_fused.json``/``poses_fused/``/``track_metrics.json`` go.
    Defaults to the stage directory (alongside ``poses.json``)."""

    gate: JumpGateConfig = JumpGateConfig()
    """Jump gate parameters; defaults are calibrated on the yellow_spoon clip."""


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = tyro.cli(TemporalFuseArgs)
    if args.gate is None:
        args.gate = JumpGateConfig()

    manifest, frames = load_poses(args.poses_json)
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path(args.poses_json).expanduser().resolve().parent
    )
    fused_dir = out_dir / "poses_fused"
    fused_dir.mkdir(parents=True, exist_ok=True)

    stats, dr_gate, dt_gate = gate_steps(frames, args.gate)
    fused_f, verdicts = fuse_track(frames, args.gate)

    before = track_metrics(frames)
    after = track_metrics(fused_f)
    stability_before = pose_stability_metrics(frames)
    stability_after = pose_stability_metrics(fused_f)

    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.action] = counts.get(v.action, 0) + 1
    logger.info(
        "[temporal] frames={} jumps={} dr_gate={:.1f}deg dt_gate={:.1f}mm actions={}",
        len(frames),
        sum(1 for s in stats if s.is_jump),
        dr_gate,
        dt_gate * 1000,
        counts,
    )

    entries: list[dict] = []
    verdict_by_index = {v.frame_index: v for v in verdicts}
    for f in fused_f:
        v = verdict_by_index[f.index]
        pose_filename = None
        if f.pose is not None:
            pose_filename = f"{f.index:06d}.txt"
            np.savetxt(fused_dir / pose_filename, np.asarray(f.pose, dtype=np.float64))
        entries.append(
            {
                "index": f.index,
                "pose_filename": pose_filename,
                "tracked": f.pose is not None,
                "method": f.method,
                "direction": f.direction,
                "fuse_action": v.action,
                "pivot_a": v.pivot_a,
                "pivot_b": v.pivot_b,
                "slerp_t": v.slerp_t,
                "note": v.note,
            }
        )

    manifest_out = dict(manifest)
    manifest_out["fused_from"] = str(Path(args.poses_json).expanduser().resolve())
    manifest_out["jump_gate"] = {
        "dr_gate_deg": dr_gate,
        "dt_gate_m": dt_gate,
        "config": {
            "rot_n_deg": args.gate.rot_n_deg,
            "rot_min_deg": args.gate.rot_min_deg,
            "trans_n_m": args.gate.trans_n_m,
            "trans_min_m": args.gate.trans_min_m,
        },
    }
    manifest_out["verdict_counts"] = counts
    manifest_out["entries"] = entries
    _write_json(out_dir / "poses_fused.json", manifest_out)

    _write_json(
        out_dir / "track_metrics.json",
        {
            "before": before,
            "after": after,
            "stability_before": {
                k: v for k, v in stability_before.items() if k != "per_frame"
            },
            "stability_after": {
                k: v for k, v in stability_after.items() if k != "per_frame"
            },
        },
    )
    logger.info(
        "[temporal] wrote {} and track_metrics.json", out_dir / "poses_fused.json"
    )


if __name__ == "__main__":
    main()
