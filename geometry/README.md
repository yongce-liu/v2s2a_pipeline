# geometry

MoGe monocular geometry estimation (point map / depth / normals / intrinsics)
for the `v2r_pipeline`. Two modes:

- **single** — estimate geometry for one image.
- **video** — frame-by-frame geometry for a whole video, reading the
  `process` stage's `frames.json` (the model is loaded once and reused for
  every frame).

The model itself is installed from a commit-pinned upstream Git dependency;
pipeline-specific behavior lives in `geometry/moge_model.py`.

## Install

```bash
uv sync
```

Requires CUDA >= 13.0 (PyTorch cu130 wheels — MoGe's own `pyproject.toml`
pins its torch index to cu130, so this env follows it rather than segment's
cu128).

## Checkpoint

The default checkpoint is `weights/moge-3/moge-3-vitg/model.pt` — the
largest **MoGe-3 (ViT-giant)** checkpoint, with metric scale and the sparse
refiner (`refine-steps`, default 3). The older v2 checkpoint still lives at
`weights/moge/model.pt` (`weights/` is a symlink to the shared do-as-i-do
weights); pass `--moge.version v2 --moge.checkpoint weights/moge/model.pt`
to use it. Other MoGe-3 sizes (`moge-3-vitl`) or Hugging Face weights work
via `--moge.checkpoint` / `--moge.allow-hf-download`.

## Usage

### Single image

```bash
uv run python -m geometry.cli --command single \
  --single.image-path frame.png \
  --single.output-dir out
```

Writes into `out/`: `depth.exr`, `mask.png`, `points.npy`,
`intrinsics.npy`, `pointcloud.ply`, a colorized `vis/000000_depth_vis.png`,
and a `result.json` summary.

### Video (frame-by-frame)

```bash
uv run python -m geometry.cli --command video \
  --video.frames-json outputs/0/process/frames.json \
  --video.vis
```

For `outputs/0/process/frames.json` this creates (mirroring the `segment`
stage layout):

```
outputs/0/geometry/
├── config.json      # effective run config (same style as process/segment)
├── geometry.json    # per-frame manifest (paths to every artifact)
├── frames/
│   ├── 000000/
│   │   ├── depth.exr        # float32 depth map
│   │   ├── mask.png         # valid-pixel mask (0/255)
│   │   ├── points.npy       # camera-space point map (H, W, 3), float32
│   │   ├── intrinsics.npy   # denormalized 3x3 camera intrinsics (pixel units)
│   │   └── pointcloud.ply   # only when --video.save-ply is on
│   └── ...
└── vis/             # only when --video.vis is on
    ├── 000000_depth_vis.png
    └── ...
```

(The smoke-test artifacts under `geometry/outputs/` were written before the
default was aligned with `process`/`segment`; fresh runs land in the
top-level `outputs/`.)

Downstream stages read `geometry.json` rather than guessing filenames. The
denormalized intrinsics follow the do-as-i-do convention consumed by e.g.
HaWoR: `fx = K[0,0] * W`, `fy = K[1,1] * H`, `cx = K[0,2] * W`,
`cy = K[1,2] * H`.

## Options (video mode)

- `--video.frames-json <path>` — path to the `process` stage `frames.json`.
- `--video.output-root <dir>` — root under which `<clip>/geometry/` is created.
- `--video.vis` / `--video.no-vis` — write a colorized depth visualization per
  frame (default on).
- `--video.max-frames <N>` — limit the number of frames processed.
- `--video.save-ply` — also write a colored `pointcloud.ply` per frame.
- `--video.moge.checkpoint <path>` — MoGe checkpoint
  (default `weights/moge-3/moge-3-vitg/model.pt`).
- `--video.moge.*` — other MoGe settings: `version` (`v1`/`v2`/`v3`),
  `allow-hf-download`, `device`, `fov-x`, `resolution-level`, `num-tokens`,
  `refine-steps` (v3 only), `use-fp16` / `no-use-fp16`, `force-projection`,
  `apply-mask` / `no-apply-mask`, `overwrite`.
