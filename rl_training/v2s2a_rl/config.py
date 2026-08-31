"""Runtime configuration shared by CLI and Isaac Lab task registration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeTaskConfig:
    bundle: Path
    output_dir: Path


def set_runtime_paths(bundle: Path, output_dir: Path) -> None:
    """Publish paths before Gym resolves the registered environment config."""
    os.environ["V2S2A_RL_BUNDLE"] = str(bundle.expanduser().resolve())
    os.environ["V2S2A_RL_OUTPUT_DIR"] = str(output_dir.expanduser().resolve())


def bundle_from_env() -> Path:
    value = os.environ.get("V2S2A_RL_BUNDLE")
    if not value:
        raise RuntimeError("V2S2A_RL_BUNDLE is unset; launch through `v2s2a-rl train|eval`")
    return Path(value).expanduser().resolve()
