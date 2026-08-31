#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/model_N.pt [video_length]" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TRIAL="$REPO/outputs/yellow_spoon/scene_construction/sharpa/left/yellow_spoon/0"
CHECKPOINT="$(realpath "$1")"
VIDEO_LENGTH="${2:-100}"

cd "$HERE"
export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y
uv run --no-sync v2s2a-rl eval \
  --bundle "$TRIAL/rl/task_bundle.json" \
  --output-dir "$REPO/outputs/yellow_spoon/rl_evaluation" \
  --checkpoint "$CHECKPOINT" \
  --num-envs 1 \
  --episodes 1 \
  --video \
  --video_length "$VIDEO_LENGTH"

echo "Video directory: $(dirname "$CHECKPOINT")/videos/eval"
