#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TRIAL="$REPO/outputs/yellow_spoon/scene_construction/sharpa/left/yellow_spoon/0"

cd "$HERE"
uv run --no-sync v2s2a-rl prepare \
  --trajectory "$TRIAL/trajectory_kinematic.npz" \
  --scene "$TRIAL/scene.xml" \
  --keypoints "$REPO/outputs/yellow_spoon/scene_construction/mano/left/yellow_spoon/0/trajectory_keypoints.npz" \
  --output "$TRIAL/rl/task_bundle.json" \
  --name yellow_spoon
