# physics_opt

Stage 5 of the do-as-i-do retargeting pipeline: sampling-based MPC physics
optimization with MuJoCo Warp.

Reads `scene.xml` + `trajectory_kinematic.npz` (from `retarget`) and optimizes a
physically-consistent hand + object trajectory, writing `trajectory_mjwp.npz`
(with per-step tracking-error metrics) and the resolved `config.yaml`.

Requires an NVIDIA GPU with CUDA (MuJoCo Warp runs on GPU), and a browser for
the default viser viewer.

```bash
uv sync
uv run physics_opt --task whisking
# headless: --no-show-viewer --no-wait-on-finish; bound length: --max-sim-steps 500
```

Optimizer defaults live in `config/default.yaml` (plus
`config/override/do_as_i_do.yaml`), layered with CLI overrides in
`physics_opt.workflow.load_mjwp_config` — same wiring as do-as-i-do's
`launch.py` stage 5.
