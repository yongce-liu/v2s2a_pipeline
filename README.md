# v2sim2a_pipeline
## Retargeting workspace (do-as-i-do split)

The do-as-i-do `retargeting/` pipeline is decomposed into three sibling
packages under this repo, each with its own uv env (same style as
`process/`/`segment/`):

| package | do-as-i-do stage | CLI |
|---|---|---|
| `hand_object_alignment/` | optional validated camera-frame correction of object poses against hands (auto fit or manual override) | `uv run hand_object_alignment --clip-root outputs/yellow_spoon --mode auto_per_frame` |
| `scene_construction/` | 1. dataset processing, 2. CoACD convex decomp, 3. MuJoCo scene generation | `uv run scene_construction --clip-root outputs/yellow_spoon --object-trajectory auto` |
| `retarget/` | 4. mink IK, 4.5. pedestal resolution | `uv run retarget --task yellow_spoon` |
| `physics_opt/` | 5. sampling-based MPC physics optimization (MuJoCo Warp) | `uv run physics_opt --task yellow_spoon` |

All three write into the clip root `outputs/<clip>/`: scene_construction
defaults its output root to `outputs/<clip>/scene_construction` (do-as-i-do
`{robot}/{hand}/{task}/{data_id}` layout + `assets/` inside), retarget and
physics_opt point at the same directory by default, and stage-5 artifacts land
in `.../{robot}/{hand}/{task}/{data_id}/physics_opt/`. Robot-hand MJCF assets
live in the repo-level `assets/hands/` directory (untracked — managed outside
git).

`scene_construction` reads the v2s2a stage outputs under `--clip-root`
(`process/hand_recon/segment/geometry/pose_estimation`) directly — gravity is
estimated with GeoCalib (`scene_construction.gravity`, cached under
`<clip>/scene_construction/gravity.json`) and the metric mesh scale comes
from the obj_recon layout (`pose_estimation` applies it before tracking).
`--object-trajectory auto|canonical|aligned` controls the optional
`hand_object_alignment` trajectory override; canonical pose-estimation
metadata and raw outputs remain authoritative and unchanged.

The bundled whisking example (`inputs/whisking`, outputs under
`outputs/{mano,sharpa,assets}`) is byte-identical to do-as-i-do's run and can
be replayed with `physics_opt`'s viewer or `viser`.
