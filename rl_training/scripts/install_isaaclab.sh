#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ISAACLAB_COMMIT="28a37cecdd433c22d9eabd6a5954add9f13a8951" # v3.0.0-beta2
ISAACLAB_SRC="${ISAACLAB_SRC:-$ROOT/.deps/IsaacLab}"

# Isaac Lab 3.0.0 beta2 / Isaac Sim 6.x require Python 3.12.
uv venv --python 3.12 --seed .venv

# Install Isaac Sim and the RL runtime through uv using NVIDIA's index.
uv pip install --python .venv/bin/python -e '.[isaaclab]' \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  --prerelease=allow

# beta2's setuptools metadata does not package extension.toml/package data
# correctly when each Git subdirectory is built as an isolated wheel. Keep one
# immutable source checkout and install the required extensions editable, which
# is also the official Isaac Lab external-project development layout.
if [[ ! -d "$ISAACLAB_SRC/.git" ]]; then
  mkdir -p "$(dirname "$ISAACLAB_SRC")"
  git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_SRC"
fi
git -C "$ISAACLAB_SRC" fetch --depth 1 origin "$ISAACLAB_COMMIT"
git -C "$ISAACLAB_SRC" checkout --detach "$ISAACLAB_COMMIT"

for extension in isaaclab isaaclab_assets isaaclab_physx isaaclab_rl isaaclab_tasks; do
  uv pip install --python .venv/bin/python --no-deps -e "$ISAACLAB_SRC/source/$extension"
done

OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y .venv/bin/python - <<'PY'
import isaaclab
import isaaclab_rl
import isaaclab_tasks
print("Isaac Lab:", isaaclab.__version__)
print("Isaac Lab source import OK")
PY

echo "Installed v3.0.0-beta2 source at $ISAACLAB_SRC"
