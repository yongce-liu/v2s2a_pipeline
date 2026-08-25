# scene_construction

Stages 1-3 of the do-as-i-do retargeting pipeline: turn the v2s2a clip outputs
(MANO hand tracks + object pose trajectory + object mesh) into a MuJoCo
manipulation scene.

| stage | module | output |
|---|---|---|
| 1. preprocess (clean + gravity-align) | `scene_construction.pipeline.process_dataset` | `outputs/mano/{hand}/{task}/0/trajectory_keypoints.npz` |
| 2. convex decompose (CoACD) | `scene_construction.pipeline.decompose_mesh` | `outputs/assets/objects/{obj}/convex/*.obj` |
| 3. scene generation | `scene_construction.pipeline.generate_scene` | `outputs/{robot}/{hand}/{task}/0/scene_ik.xml` |

Input contract: a v2s2a clip root (`outputs/<clip>/`) containing the upstream
stage outputs — `process/` (frames), `hand_recon/` (MANO track), `segment/`
(masks), `geometry/` (metric point maps + intrinsics), `pose_estimation/`
(metric object poses) and `obj_recon/` (SAM3D mesh + layout). Gravity is
estimated with GeoCalib (`scene_construction.gravity`, a git submodule at
`pkgs/GeoCalib` installed editable) and cached under
`<clip>/scene_construction/gravity.json`; `--gravity up|json` skips the model.
The metric mesh scale comes from the obj_recon `layout.json` (the same scale
`pose_estimation` applies before tracking). An optional validated
`hand_object_alignment/poses.json` can override only the trajectory. Selection
is explicit with `--object-trajectory auto|canonical|aligned`; `auto` preserves
legacy poses when the optional stage is absent, disabled, or rejected. Use
`--alignment-manifest PATH` to select another compatible manifest.

Robot-hand MJCF assets (sharpa) live in the repo-level `assets/hands/`
directory (untracked — managed outside git); the package only references them
via `scene_construction.paths`

```bash
uv sync
uv run scene_construction --clip-root ../outputs/yellow_spoon
```

Outputs keep the do-as-i-do `outputs/` layout, so the `retarget` and
`physics_opt` packages (and the original do-as-i-do repo) can consume them
interchangeably.
