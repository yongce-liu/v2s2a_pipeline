# hand_recon

HaWoR-based hand mesh reconstruction for the `v2s2a_pipeline`. Wraps the
upstream `pkgs/HaWoR` scripts behind the same stage conventions as `process`
and `segment`: reads a clip's `frames.json` from `process`, writes its
outputs under `outputs/<clip>/hand_recon/`.

## Install

```bash
scripts/apply_patches.sh   # once after cloning or updating submodules
cd hand_recon
uv sync
```

Requires CUDA >= 12.8 (PyTorch cu128 wheels). The HaWoR checkout at
`pkgs/HaWoR` is installed as an editable hatchling wheel (`hawor`), so any
upstream edits are visible without a reinstall. `pkgs/HaWoR` itself stays an
official upstream checkout; its local packaging shim lives in
`patches/HaWoR.patch` and is applied by `scripts/apply_patches.sh`.

Run the patch script before `uv sync`: the HaWoR patch supplies the
`pyproject.toml` that uv needs while resolving the editable dependency. uv does
not currently provide a supported post-sync hook in `pyproject.toml`.

## Usage

```bash
uv run python -m hand_recon.cli \
  --frames-json outputs/0/process/frames.json
```

For `outputs/0/process/frames.json` this creates:

```
outputs/0/hand_recon/
├── config.json       # effective run config (same style as process/segment)
├── hands.json        # stage metadata + published artifact pointers
├── hand_anchors.json # per-frame HaWoR left/right 2D boxes and center points
├── meshes.npz        # per-frame MANO vertices/joints for left/right
└── vis_0_-1/
    ├── aitviewer/    # per-frame overlay renders
    └── overlay.mp4
```

## Notes

- `meshes.npz` mirrors the layout do-as-i-do consumers expect
  (`left_vertices`, `left_joints`, `left_faces`, `left_trans`, `left_rot`,
  `left_hand_pose`, `left_betas`, `left_valid`, and the right-hand twins).
- `hand_anchors.json` publishes HaWoR detector results independently of the
  scratch workspace. Each frame contains `left hand` and `right hand` entries
  with pixel-space `box_xyxy`, box-center `point_xy`, detection confidence,
  handedness score, and track ID. Missing detections are represented by `null`.
- HaWoR hard-codes relative paths to its assets (`./weights`, `./_DATA`,
  `thirdparty/...`); the stage mounts those over a scratch workspace under
  `outputs/<clip>/hand_recon/.hawor_runtime/` with symlinks, so the source
  checkout is never touched. Broken symlinks are pruned on exit.
- The wrapper never edits files under `pkgs/HaWoR`; the only behavioral
  deviations (e.g. headless aitviewer, daid-style PNG frames) arrive as
  subclasses and monkey-patches inside `hand_recon`.

## Options

- `--frames-json <path>` — path to the `process` stage `frames.json`.
- `--output-root <dir>` — root under which `<clip>/hand_recon/` is created
  (default `outputs`, same convention as `process`/`segment`).
- `--checkpoint` / `--infiller-weight` — HaWoR weights (default resolved from
  `pkgs/HaWoR/weights/hawor/checkpoints/`).
- `--img-focal <px>` — image focal length; omit to reuse what HaWoR already
  estimates per video.
- `--static-camera` / `--no-static-camera` — skip DROID-SLAM and treat the
  camera as fixed (default on).
- `--vis` / `--no-vis` — render the aitviewer overlay (default on).
- `--max-frames <N>` — cap the frames processed for smoke tests.

Downstream geometry stages should read `meshes.npz`; segmentation stages should
read `hand_anchors.json`. Neither should depend on paths inside
`.hawor_runtime/`.
