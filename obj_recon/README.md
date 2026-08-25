# obj_recon

SAM 3D Objects mesh reconstruction for the `v2s2a_pipeline`. It joins the
`segment` stage's `masks.json` with precomputed point maps from the `geometry`
stage's `geometry.json`, then writes one mesh per (frame, prompt) into a sibling
`obj_recon/` stage directory. SAM3D's internal MoGe model is disabled: geometry
is estimated once upstream rather than recomputed here.

Two modes are supported:

- **single-view (default)** — one canonical mesh per selected frame;
- **multi-view (`--mv.enabled`, MV-SAM3D)** — fuse several keyframes into one
  higher-fidelity mesh per object plus a view-0 metric pose, reusing the same
  geometry point maps (no DA3). See [Multi-view mode](#multi-view-mode-mv-sam3d).

## Install

```bash
cd obj_recon
uv sync
```

**Requirements:** Python 3.11 (matches the prebuilt Blackwell wheels),
CUDA >= 12.8 (PyTorch cu128 wheels for RTX 5090 / sm_120).

`sam-3d-objects` is installed as an editable path dependency from
`../pkgs/MV-SAM3D` (the multi-view fork). It declares the SAME project name
(`sam3d_objects`) and top-level package as upstream. MV-SAM3D is a superset of
upstream (adds
`sam3d_objects.pipeline.multi_view_weighted`, `multi_view_utils`,
`pose_align/`) with unchanged single-frame behavior, so both the single-view
and multi-view paths run against it. Several heavy CUDA extensions are pinned to
**prebuilt wheels** matching the daid-sam3d env (torch 2.8.0+cu128, py3.11,
Blackwell) — see `[tool.uv.sources]` in `pyproject.toml`:

| Package | Source |
|---|---|
| `pytorch3d==0.7.9+d9839a9pt2.8.0cu128` | MiroPsota torch_packages_builder |
| `nvdiffrast==0.4.0` | HF zerogpu-blackwell-wheels (pt28-cu128-cp311) |
| Gaussian rasterization | `gsplat>=1.5.3` from PyPI (default backend; no local rasterizer wheel) |
| `kaolin==0.18.0` | NVIDIA S3 wheel (`torch-2.8.0_cu128`) |
| `spconv_cu120==2.3.6`, `xformers`, `gsplat`, etc. | PyPI |

MV-SAM3D's `requirements.txt` hard-pins versions that
are incompatible with cu128 / RTX 5090 (e.g. `torch==2.5.1+cu121`,
`kaolin==0.17.0`, `spconv-cu121`), so `obj_recon` relaxes them via
`[tool.uv] override-dependencies`. SAM3D still imports a few MoGe 1.x geometry
helper functions, but its model is neither constructed nor executed.

The default pipeline config is `weights/sam3d/pipeline.yaml`; its referenced
SAM3D checkpoints must live alongside it in `weights/sam3d/`.

**Blackwell note:** `XFORMERS_DISABLED=1` selects the SDPA attention backend on
RTX 50-series. The package sets the upstream `LIDRA_SKIP_INIT` compatibility
flag internally, so no Conda environment is required.

## Usage

```bash
export XFORMERS_DISABLED=1

geometry/.venv/bin/python -m geometry.cli --command video \
  --video.frames-json outputs/yellow_spoon/process/frames.json

obj_recon/.venv/bin/python -m obj_recon.cli \
  --masks-json outputs/yellow_spoon/segment/masks.json \
  --geometry-json outputs/yellow_spoon/geometry/geometry.json \
  --prompt-id "yellow spoon" \
  --max-frames 10
```

The upstream pipeline is loaded **once** and reused across all frames.

### Which objects get reconstructed

Each `masks.json` entry carries a `prompt_masks` list — one mask per tracked
prompt (hands AND scene objects). By default obj_recon picks every prompt
whose `input_type == "text"` (i.e. the scene objects — hands already have
meshes from `hand_recon`):

```
--prompt-id "yellow spoon"              # only the spoon
--prompt-id "yellow spoon" --prompt-id "left hand"   # several
# (omit --prompt-id)                    # every text-prompt object
```

### Input contract

- `masks_json` — `outputs/<clip>/segment/masks.json` (required)
- `geometry_json` — `outputs/<clip>/geometry/geometry.json` (required); entries
  are joined to segment entries by frame `index`; each `points.npy` must be
  float `(H, W, 3)` and each `intrinsics.npy` must be a pixel-space `3x3`
  matrix for the source-frame resolution
- `frames_json` — optional; if omitted, resolved via the
  `source_frames_json` field in `masks.json`, falling back to
  `../process/frames/`
- Each entry must contain `prompt_masks[].mask_filename` relative to
  `masks.json`'s `masks_dir`

### Options

- `--masks-json <path>` — segment stage's mask manifest (**required**).
- `--geometry-json <path>` — geometry stage's point-map manifest (**required**).
- `--frames-json <path>` — process stage's frame manifest (optional).
- `--output-root <dir>` — root under which `<clip>/obj_recon/` is created
  (default: the segment clip root).
- `--prompt-id <str>` — repeat to select specific prompts.
- `--max-frames <int>` — cap frames processed (smoke tests).
- `--skip-existing` — skip frames that already have a `layout.json`.
- `--recon.*` — SAM 3D Objects settings: `config`, `device`, `seed`,
  `with-mesh-postprocess`, `with-texture-baking`, and `use-vertex-color`.

## Output structure

Given `outputs/yellow_spoon/segment/masks.json`, outputs land in
`outputs/yellow_spoon/obj_recon/`:

```
outputs/yellow_spoon/obj_recon/
├── meshes/
│   ├── 000000/                 # one folder per processed frame
│   │   ├── yellow_spoon/
│   │   │   └── yellow_spoon.obj
│   │   └── layout.json         # per-object local→scene transforms
│   ├── 000001/
│   │   └── ...
│   └── ...
├── meshes.json                 # per-frame mesh manifest
└── config.json                 # effective run configuration
```

`layout.json` follows the SAM3D `make_scene` convention: `translation`,
uniform `scale`, `quat_wxyz`, plus a procrustes-aligned `new_quat` that maps
the mesh from its canonical Y-up frame to the ISAAC frame. Downstream stages
(guided pose tracking, scene export) read these files.

## Multi-view mode (MV-SAM3D)

`--mv.enabled` switches to MV-SAM3D (vendored at `pkgs/MV-SAM3D`): instead of a
single-view reconstruction it fuses several keyframes into **one** mesh per
object with entropy-weighted attention fusion — generally a more faithful mesh
and pose than any single view. It reuses the **same** `geometry` stage point
maps (no DA3 / Depth-Anything-3 dependency).

```bash
# default strategy: 4 evenly-spaced keyframes
obj_recon/.venv/bin/python -m obj_recon.cli \
  --masks-json outputs/yellow_spoon/segment/masks.json \
  --geometry-json outputs/yellow_spoon/geometry/geometry.json \
  --mv.enabled

# pick keyframes by hand (first = view-0 reference pose); NOTE: repeat the
# values under ONE flag, not the flag itself — tyro keeps only the last
# occurrence of a repeated list flag.
obj_recon/.venv/bin/python -m obj_recon.cli \
  --masks-json outputs/yellow_spoon/segment/masks.json \
  --geometry-json outputs/yellow_spoon/geometry/geometry.json \
  --mv.enabled --mv.keyframe-strategy manual --frame-index 0 44 88

# ffprobe scene/I-frames as keyframes (falls back to even spacing)
  --mv.enabled --mv.keyframe-strategy ffprobe --mv.keyframe-video inputs/yellow_spoon.mp4
```

Options (all under `--mv.*`): `keyframe-strategy` (`even`|`manual`|`ffprobe`),
`num-views` (default 4), `max-views-cap` (default 8), `seed`, weighting knobs.
Use `--recon.with-texture-baking=true` for the textured GLB.

Both modes share the camera / intrinsics convention: geometry supplies MoGe
point maps + intrinsics; MV-SAM3D's external-pointmap branch would otherwise
discard the known intrinsics and re-run MoGe's `recover_focal_shift`, which
NaNs on MoGe's own point maps, so `multiview.py` injects the geometry
intrinsics per view (checked by `tests/test_multiview.py`-adjacent behavior).

### Outputs

One folder per object under `meshes/mv/<object>/`:

```
meshes/mv/yellow_spoon/
├── yellow_spoon.glb        # fused, textured canonical mesh
├── yellow_spoon.obj + material_0.png
├── view_poses.json         # view-0 reference pose (+ per-view poses when reported)
└── layout.json             # view-0 metric pose — same schema pose_estimation consumes
```

`meshes/mv/<object>/layout.json` is intentionally **schema-compatible** with
the single-frame `layout.json`, so `pose_estimation --mesh-path
outputs/<clip>/obj_recon/meshes/mv/yellow_spoon/yellow_spoon.obj` consumes the
fused mesh + scale exactly as before; intermediate keyframe poses are then
filled by pose_estimation's tracking.
