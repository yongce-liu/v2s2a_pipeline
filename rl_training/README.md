# v2s2a RL training

`rl_training/` turns the assets and trajectories produced by the existing
v2s2a pipeline into a robust dexterous-manipulation policy.

The implementation uses:

- **Isaac Lab 3.0.0 beta2 / Isaac Sim 6.x** (Python 3.12);
- **RSL-RL PPO** through Isaac Lab's public vector-environment wrapper;
- the existing `scene.xml` as the authoritative robot + reconstructed object +
  support/scene asset, converted once to USD with NVIDIA's
  `mujoco-usd-converter` (preserving convex collision meshes and dynamics);
- the retargeted `trajectory_kinematic.npz`, split into hand and object targets;
- residual actions, random phase resets, future-reference observations,
  decaying virtual object assistance, hand-keypoint and wrist-object relative-pose
  tracking, contact rewards, early termination and
  task-level success evaluation.

This follows the architecture of `video_to_data.new/robotic_grounding` while
adapting its data contract to v2s2a's MuJoCo/NPZ outputs. The simulator,
training algorithm, wrappers and policy exporters remain upstream Isaac Lab /
RSL-RL components rather than local replacements.

## Inputs

A task consumes the four explicit input classes requested by the pipeline:

| input | bundle source |
|---|---|
| hand trajectory | first `D-7` columns of `trajectory_kinematic.npz:qpos` |
| object trajectory | final free-joint pose `[xyz, qwxyz]` |
| robot/hand asset | generated scene's robot asset tree |
| object + scene assets | `scene.xml` and every referenced convex/visual mesh/support |
| hand keypoints/contact schedule | `trajectory_keypoints.npz`; proximity fallback when explicit custom contact labels are empty |

`prepare` validates that the MJCF joint layout and trajectory dimensions agree,
converts the scene to a reusable layered USD, and writes a checksummed
`task_bundle.json`. The converter lacks MJCF explicit-pair support, so the
pair allow-list is baked into collision masks before conversion. Six Cartesian
wrist joints are represented as the hand articulation's floating root while the
22 finger joints remain PD-controlled.

## Install with uv

The beta requires Python 3.12. Isaac Sim/runtime dependencies are encoded in
`pyproject.toml`; the exact beta2 source checkout is pinned by the install script:

```bash
cd rl_training
bash scripts/install_isaaclab.sh
```

The script follows the beta guide's index strategy while installing the exact
`v3.0.0-beta2` commit (SHA pinned in the script), because the beta's monolithic
`isaaclab` wheel is not currently published on public PyPI. The beta source
extensions are installed editable from `.deps/IsaacLab` because isolated wheels
omit their extension metadata/package trees:

```bash
uv venv --python 3.12 --seed .venv
uv pip install -e '.[isaaclab]' \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  --prerelease=allow
```

Set both `OMNI_KIT_ACCEPT_EULA=Y` and `ACCEPT_EULA=Y` for non-interactive
NVIDIA EULA acceptance after reviewing its terms. The first Isaac Sim launch
can spend several minutes filling its extension cache.

> Do not replace the source pins with unversioned `isaaclab`: public PyPI
> currently resolves that name to incompatible IsaacLab 2.x wheels.

## Prepare a task

Yellow-spoon example:

```bash
bash scripts/prepare_yellow_spoon.sh
```

Generic command:

```bash
uv run --no-sync v2s2a-rl prepare \
  --trajectory /path/to/trajectory_kinematic.npz \
  --scene /path/to/scene.xml \
  --keypoints /path/to/trajectory_keypoints.npz \
  --output /path/to/rl/task_bundle.json \
  --name my_task
```

Prefer `trajectory_kinematic.npz` for a complete reference. Stage-5
`physics_opt/trajectory_mjwp.npz` is an MPC trace with batched windows rather
than one contiguous demonstration; it should first be consolidated before use.

## Train

```bash
export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y
uv run --no-sync v2s2a-rl train \
  --bundle ../outputs/yellow_spoon/scene_construction/sharpa/left/yellow_spoon/0/rl/task_bundle.json \
  --output-dir ../outputs/yellow_spoon/rl_training \
  --num-envs 2048 \
  --max-iterations 20000 \
  --viz none
```

Useful overrides are forwarded to the Isaac Lab launcher/RSL-RL runner, e.g.
`--device cuda:0`, `--logger wandb` and `--log_project_name v2s2a`. On the
current RTX 5090, measured throughput is about 85k steps/s at 8192 envs versus
65k at 16384 envs, so 8192 is the recommended throughput-optimal setting.

### Periodic training visualization

Record one rollout every 50 PPO iterations (each iteration has 32 control
steps by default):

```bash
VIDEO_EVERY_ITERATIONS=50 VIDEO_LENGTH=100 \
  bash scripts/train_with_video.sh
```

Or add the equivalent options to a training command:

```bash
--video --video_every_iterations 50 --video_length 100
```

Videos are written under the active run's `videos/train/` directory. Recording
requires rendering and reduces throughput, so short, infrequent clips are
recommended.

Training outputs:

```text
<output-dir>/logs/rsl_rl/v2s2a_trajectory_hand/<run>/
  model_*.pt
  params/{env,agent}.yaml
  summaries/events...
  videos/train/...        # when --video is enabled
```

## Evaluate and export

```bash
uv run --no-sync v2s2a-rl eval \
  --bundle .../rl/task_bundle.json \
  --output-dir ../outputs/yellow_spoon/rl_training \
  --checkpoint .../model_20000.pt \
  --num-envs 64 \
  --episodes 500 \
  --viz none
```

Evaluation runs with no teacher mixing, no virtual object assistance and frame
0 resets. To render one checkpoint interactively and save an MP4:

```bash
bash scripts/render_policy.sh /path/to/model_999.pt 100
```

The video is saved under the checkpoint run's `videos/eval/`. Evaluation also writes:

- `evaluation.json` with task completion rate and reward;
- `exported/policy.pt` (TorchScript);
- `exported/policy.onnx`.

A policy should be called “good” only after its **unassisted frame-0 success
rate** is stable across multiple seeds and domain-randomized evaluations—not
because imitation reward is high while the virtual controller is active.

## Tests

Core data-contract tests do not require Isaac Sim:

```bash
uv sync --no-default-groups --extra dev
uv run --no-sync pytest
```

A simulator smoke test should use `--num-envs 2 --max-iterations 1 --viz none`
after Isaac Lab is installed.

## Current scope and extension points

- Current upstream examples are single-hand/single-rigid-object. The bundle
  validator deliberately rejects multiple free object joints rather than
  training on an ambiguous layout.
- The current bundle contract supports one hand and one manipulated free object.
  Bimanual/two-object trajectories are rejected until the schema and environment
  represent each named hand/object explicitly; silently treating the second object
  as policy-controlled joints would be unsafe.
- Articulated and multi-object tasks should extend the bundle schema with named
  object bodies/joints and per-object success predicates instead of guessing.
- Scene-specific success predicates (drawer opened, object inside target,
  relative pose achieved) should be added beside geometric terminal tracking;
  the generic terminal object-pose criterion is the safe default.

## Roadmap: direct MANO conditioning

Direct MANO + object trajectory conditioning is intentionally deferred while the
retargeted-trajectory RL baseline is improved and benchmarked. The proposed
architecture, losses, staged migration and go/no-go criteria are documented in
[`MANO_RETARGETING_ROADMAP.md`](MANO_RETARGETING_ROADMAP.md).
