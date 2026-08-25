# hand_object_alignment

Corrective alignment between the reconstructed MANO hand trajectory and the
FoundationPose object trajectory. When `pose_estimation/poses_filtered.json`
exists it is selected by default; otherwise the raw `poses.json` is used. Raw
upstream outputs are never edited; the stage writes corrected 4x4 object-to-camera matrices under
`outputs/<clip>/hand_object_alignment/` and records whether measurable
acceptance gates passed.

```bash
uv sync
uv run hand_object_alignment \
  --clip-root ../outputs/yellow_spoon \
  --mode auto_per_frame
```

## Why not a user-selected global rigid transform

A hand-picked camera-frame rigid transform left-composed onto every pose can
only fix one *constant* offset shared by the whole clip. Real misalignment is
per-frame: HaWoR's depth scale and FoundationPose's z-bias drift as the hand
closes around the object, peak precisely during the grip phase where
downstream physics needs fidelity, and are unobservable from a single 6-DoF
parameter. That is why this stage is optional and validated, not blankly
applied: an alignment the pipeline cannot measure is more dangerous than no
alignment at all.

## Automatic correction — do-as-i-do, without a simulator

do-as-i-do's physics stage has a warmup: it pushes the object toward the
first penetration-free pose at contact distance, then accepts as soon as the
closest hand↔object distance **converges** (a distance gate, not an iteration
budget). The automatic mode here is the same idea in closed form per frame:

1. **Warmup seed.** A translation-only grid search moves the object until its
   mean Huber-clipped hand distance stops improving.
2. **6-DoF Powell polish** minimizes a *contact-only* objective: Huber
   distance from object verts within `contact_band_m` of the hand (frozen
   correspondence from the warmup), plus an L2 penetration-depth term for
   vertices inside the hand convex hull, plus a strong trust-region prior
   pinning the correction near identity. The prior is what makes a manual
   override unnecessary — the fit can only move what the evidence demands,
   and only this far.

`--mode auto_global` fits one shared correction across every in-hand frame
(do-as-i-do's trajectory-wide warmup). `--mode auto_per_frame` (default) fits
independently and is what handles per-frame drift. `--mode manual` keeps the
original hand-tuned override for ablation and rescue.

## Acceptance gates (all measurable)

The manifest is only *usable* when every gate passes:

| Gate | Default |
| --- | --- |
| median post-fit hand‑object clearance ≤ `contact_dist_m` | 2 cm |
| max post-fit penetration depth ≤ `max_penetration_m` | 5 mm |
| median post clearance ≤ median pre clearance (no regression) | on |
| |translation| ≤ `max_translation_m` (hard clamp) | 5 cm |
| |rotation| ≤ `max_rotation_deg` (hard clamp) | 15° |
| ≥ `min_inhand_overlap_frames` with pose and valid hand | 2 |
| ≥ `min_tracked_frames` corrected | 2 |

Failed gates write a rejected manifest with per-gate error strings and keep
`poses/` deleted, so downstream stages can never silently consume a bad fit.

## Outputs

```text
<clip>/hand_object_alignment/
├── config.json     # every argument that produced this run
├── poses.json      # schema 1.0 manifest
└── poses/000000.txt
```

`poses.json` carries `stage` / `status` (`accepted`, `rejected`, `disabled`)
/ `usable`, `fit_stats.aggregate` (every gate metric before/after), and
`fit_stats.per_frame` (the fitted correction and per-frame clearance record,
for audit). `scene_construction` consumes it optionally via
`--object-trajectory aligned`; `auto` falls back to the canonical
pose_estimation output whenever the manifest is missing, disabled, or
rejected. Source manifests and pose files remain untouched.
