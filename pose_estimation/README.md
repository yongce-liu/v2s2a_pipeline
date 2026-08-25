# pose_estimation

FoundationPose-based 6D object pose estimation and tracking for the
`v2s2a_pipeline`. Wraps the vendored `pkgs/FoundationPose` behind the same
stage conventions as `process` / `segment` / `geometry`: reads a clip's
`frames.json` from `process`, the object masks from `segment`'s
`masks.json`, and the per-frame depth/intrinsics from `geometry`'s
`geometry.json`, writes its outputs under `outputs/<clip>/pose_estimation/`.

## Install

```bash
cd pose_estimation
uv sync
```

Requires CUDA >= 12.8 (PyTorch cu128 wheels). The FoundationPose checkout at
`pkgs/FoundationPose` is installed as an editable setuptools wheel
(`foundationpose`), so upstream code stays unmodified.

**One-time setup** (things `uv sync` cannot express):

1. **Weights** — the refiner/scorer checkpoints live in
   `weights/foundationpose/{2023-10-28-18-33-37,2024-01-11-20-02-45}/`
   and are exposed to FoundationPose via the `pkgs/FoundationPose/weights`
   symlink.

2. **mycpp extension** (pose clustering C++/pybind11, required by the
   estimator; needs cmake + boost + eigen3 + pybind11 headers, e.g. from
   the `daid-foundationpose` conda env):

   ```bash
   cd pkgs/FoundationPose/mycpp && mkdir -p build && cd build
   env PATH=/mnt/workspace/miniforge3/envs/daid-foundationpose/bin:$PATH \
     cmake .. -DPYTHON_EXECUTABLE=$(pwd)/../../../pose_estimation/.venv/bin/python \
       -DCMAKE_PREFIX_PATH=/mnt/workspace/miniforge3/envs/daid-foundationpose
   make -j
   ```

   `pytorch3d` / `nvdiffrast` need no manual step: their prebuilt wheels
   (torch 2.8.0 + cu128 + cp312, Blackwell-compatible) come from the
   [torch_packages_builder](https://miropsota.github.io/torch_packages_builder/)
   PEP 503 index declared in `pyproject.toml`, so `uv sync` installs them
   and `uv.lock` pins them.

## Usage

```bash
uv run python -m pose_estimation.cli \
  --frames-json outputs/yellow_spoon/process/frames.json \
  --masks-json outputs/yellow_spoon/segment/masks.json \
  --mesh-path outputs/yellow_spoon/obj_recon/yellow_spoon.obj \
  --init-frame 0
```

For `outputs/yellow_spoon/process/frames.json` this creates:

```
outputs/yellow_spoon/pose_estimation/
├── config.json      # effective run config (same style as the other stages)
├── poses.json       # per-frame pose manifest (index / pose_filename / method)
└── poses/
    ├── 000000.txt   # 4x4 ob_in_cam matrix (object pose in camera frame)
    └── ...
```

For an MV mesh, the stage loads every reconstructed view from the mesh's
`view_poses.json`, chooses the temporally middle view, and seeds FoundationPose
with that view's metric object-to-camera pose. It then runs `track_one` from the
seed toward both the beginning and end of the clip. The other reconstruction
views contribute to the fused mesh but are not registration anchors, and poses
are never interpolated between them. For a single-view mesh, `--init-frame`
retains full registration followed by bidirectional tracking; `--reinit-every N`
can add periodic registrations.

## Options

- `--frames-json <path>` — path to the `process` stage `frames.json`.
- `--mesh-path <path>` — object mesh (`.obj`) to register and track.
- `--masks-json <path>` — path to the `segment` stage `masks.json` (object
  masks for registration).
- `--geometry-json <path>` — path to the `geometry` stage `geometry.json`
  (depth + intrinsics); defaults to `<clip>/geometry/geometry.json`.
- `--init-frame <N>` — reference frame in single-anchor mode (default 0).
- `--anchor-frames <N...>` — explicit registration anchors for a non-MV mesh;
  ignored for an MV mesh, which always uses the middle `view_poses.json` view.
  MV tracking requires this versioned `view_poses.json` contract next to the mesh.
- `--prompt-id <name>` — per-object segmentation prompt used for registration;
  defaults to the mesh/object name with underscores replaced by spaces.
- `--reinit-every <N>` — optionally add periodic registrations (default 0).
- `--translation-prior` and `--anchor-translation-to-geometry` — use the object
  mask's metric point-map center to keep fast motion and depth drift bounded;
  rotation remains a FoundationPose estimate.
- `--foundationpose.track-refine-iter` (default 10) and
  `--foundationpose.track-crop-ratio` (default 2.0) — tracking quality controls.
- `--max-frames <N>` — cap the frames processed (smoke tests).
- `--output-root <dir>` — root under which `<clip>/pose_estimation/` is
  created (default `outputs`).
- `--foundationpose.*` — model settings: `weights-root`, `device`,
  `est-refine-iter` (default 5), `track-refine-iter` (default 10),
  `track-crop-ratio` (default 2.0), `debug`, `debug-dir`.
- `--temporal-filter.enabled` — run a constant-velocity error-state EKF over
  translation, velocity, and SO(3) rotation after tracking. Mask/depth quality
  scales the measurement noise, and chi-square innovation gating rejects
  symmetric-spoon flips while coasting instead of interpolating. It writes
  `poses_filtered/` and `poses_filtered.json`, preserving the raw poses.

## Notes

- FoundationPose needs **metric depth**; this stage derives it from the
  `geometry` stage's point map Z channel (invalid/inf pixels zeroed), the
  same convention do-as-i-do's `pointmap_to_depth` uses.
- `pkgs/FoundationPose` is never edited. Pipeline-specific behavior lives in
  `pose_estimation.foundationpose_estimator` (lazy imports, weight checks, a
  `register()`/`track()`/`set_pose()` API mirroring do-as-i-do's
  `track_object_foundationpose.py` calling pattern).

Downstream stages should read `poses.json` + the `poses/*.txt` matrices
rather than re-deriving paths.
