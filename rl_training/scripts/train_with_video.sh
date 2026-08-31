#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TRIAL="$REPO/outputs/yellow_spoon/scene_construction/sharpa/left/yellow_spoon/0"
OUTPUT="${OUTPUT_DIR:-$REPO/outputs/yellow_spoon/rl_training_8192_video}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
VIDEO_EVERY_ITERATIONS="${VIDEO_EVERY_ITERATIONS:-50}"
VIDEO_LENGTH="${VIDEO_LENGTH:-100}"

cd "$HERE"
export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y
uv run --no-sync v2s2a-rl train \
  --bundle "$TRIAL/rl/task_bundle.json" \
  --output-dir "$OUTPUT" \
  --num-envs "$NUM_ENVS" \
  --max-iterations "$MAX_ITERATIONS" \
  --seed 42 \
  --video \
  --video_every_iterations "$VIDEO_EVERY_ITERATIONS" \
  --video_length "$VIDEO_LENGTH" \
  --viz none
