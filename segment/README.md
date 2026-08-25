# segment

SAM3 hand-mask segmentation for the `v2r_pipeline`. Two modes:

- **single** — segment one image.
- **video** — frame-by-frame segmentation of a whole video, reading the
  `process` stage's `frames.json` (the model is loaded once and reused for
  every frame).

## Install

```bash
uv sync
```

Requires CUDA >= 12.8 (PyTorch cu128 wheels).

## Usage

### Single image

```bash
uv run python -m segment.cli --command single \
  --single.image-path frame.png \
  --single.output-dir out \
  --single.sam-mask.checkpoint ckpts/sam3/sam3.pt
```

Writes `hand_seg.png` (mask) and `hand_seg_vis.jpg` (overlay) into `out/`.

### Video (frame-by-frame)

```bash
uv run python -m segment.cli --command video \
  --video.frames-json outputs/0/process/frames.json \
  --video.sam-mask.checkpoint ckpts/sam3/sam3.pt \
  --video.sam-mask.text-prompts "human hand" "robot arm" \
  --video.vis
```

To combine geometric anchors with text prompts, pass one producer-independent
anchor manifest:

```bash
uv run python -m segment.cli --command video \
  --video.frames-json outputs/yellow_spoon/process/frames.json \
  --video.anchors-json outputs/yellow_spoon/hand_recon/hand_anchors.json \
  --video.sam-mask.checkpoint weights/sam3/sam3.pt \
  --video.sam-mask.text-prompts "yellow spoon" \
  --video.vis
```

The JSON's ordered `targets` list supplies each anchored output name and whether
it uses a `box` or `point`; no producer-specific CLI options are needed. Text
prompts contain only additional semantic objects. If a target name appears in
both places, the anchor definition wins and the duplicate text prompt is removed.

For `outputs/0/process/frames.json` this creates (mirroring the `process`
stage layout):

```
outputs/0/segment/
├── config.json      # effective run config (same style as process)
├── masks.json       # per-frame mask manifest (index / paths / bbox / area)
├── masks/
│   ├── 000000.png           # union of all prompts (backward-compatible)
│   ├── left hand/
│   │   ├── 000000.png       # "left hand" prompt only
│   │   └── ...
│   └── yellow spoon/
│       ├── 000000.png       # "yellow spoon" prompt only
│       └── ...
└── masks_vis/               # only when --video.vis is on
    ├── 000000.jpg            # colored overlay + prompt legend
    └── ...
```

`masks.json` has a top-level `prompts` list containing each prompt's stable ID,
text, overlay color, and mask directory. Each frame entry has a corresponding
`prompt_masks` list with that prompt's mask filename, instance count, bounding
box, and pixel area. The existing frame-level fields and `masks/000000.png`
remain the union across all prompts for downstream compatibility.

## Options (video mode)

- `--video.frames-json <path>` — path to the `process` stage `frames.json`.
- `--video.output-root <dir>` — root under which `<clip>/segment/` is created
  (default `outputs`, same convention as `process`).
- `--video.vis` / `--video.no-vis` — write original frame + mask overlay images
  (default on).
- `--video.max-frames <N>` — limit the number of frames processed.
- `--video.anchors-json <path>` — generic per-frame anchor manifest. Its
  `targets` array defines output IDs and `box`/`point` prompting behavior.
- `--video.sam-mask.text-prompts <prompt...>` — additional text-prompt targets
  (default `"human hand and arm"`). SAM3 computes each frame's image embedding
  once per input type. Up to eight combined targets are supported so every
  overlay category keeps a unique, fixed color.
- `--video.sam-mask.*` — other SAM3 settings: `checkpoint`,
  `allow-hf-download`, `device`, `score-threshold`, `overlay-alpha`,
  `mask-color-rgb` (first prompt), `overwrite`. The legacy `text-prompt` option
  is still accepted for single-prompt calls.

Downstream stages read `masks.json` (per-frame mask paths + positions) rather
than guessing filenames.
