# vis

Browser-based replay and debugging for `v2s2a_pipeline` outputs. The package
reads artifacts from the unified `outputs/<clip>/` tree and does not generate a
second visualization-specific data format.

## Install

```bash
cd vis
uv sync
```

## Scene replay

Replays `scene.xml` with the optimized `trajectory_mjwp.npz` (or the IK
trajectory when physics optimization has not run). When available, the viewer
also shows the blue IK ghost and orange MANO/object reference on the same
timeline.

```bash
uv run vis --clip-root ../outputs/yellow_spoon --mode scene
```

The default inputs are discovered under:

```text
outputs/<clip>/scene_construction/<robot>/<embodiment>/<task>/<id>/scene.xml
outputs/<clip>/scene_construction/<robot>/<embodiment>/<task>/<id>/trajectory_kinematic.npz
outputs/<clip>/scene_construction/<robot>/<embodiment>/<task>/<id>/physics_opt/trajectory_mjwp.npz
outputs/<clip>/scene_construction/mano/<embodiment>/<task>/<id>/trajectory_keypoints.npz
```

Use `--scene.run-dir`, `--scene.scene`, `--scene.traj`, `--scene.ik`, and
`--scene.mano` to override discovery. `--no-skip-warmup` keeps physics warmup
frames visible.

## Raw reconstruction replay

Shows camera-space hand meshes, tracked object, MoGe point map, source image
frustum, and coordinate frames. Inputs are joined by each manifest's frame
`index`.

```bash
uv run vis --clip-root ../outputs/yellow_spoon --mode raw
```

This mode consumes the project's published artifacts directly:

```text
outputs/<clip>/hand_recon/hands.json
outputs/<clip>/hand_recon/meshes.npz
outputs/<clip>/pose_estimation/poses.json
outputs/<clip>/pose_estimation/poses/*.txt
outputs/<clip>/geometry/geometry.json
outputs/<clip>/geometry/frames/*/{points,intrinsics}.npy
outputs/<clip>/process/frames/*
```

Open `http://localhost:8081` after launch. Both modes expose layer toggles, a
frame slider, playback control, and adjustable FPS.
