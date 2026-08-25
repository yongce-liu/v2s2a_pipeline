# retarget

Stages 4-4.5 of the do-as-i-do retargeting pipeline: dexterous-hand IK.

| stage | module | output |
|---|---|---|
| 4. inverse kinematics (mink) | `retarget.solve_ik` | `outputs/{robot}/{hand}/{task}/0/trajectory_kinematic.npz` |
| 4.5. pedestal resolution | `retarget.resolve_pedestal` | `outputs/{robot}/{hand}/{task}/0/scene.xml` (+ `scene_eq.xml`) |

Depends on `scene_construction` (in-hand checks, IO layout) and consumes its
`scene_ik.xml` + keypoint trajectory.

```bash
uv sync
uv run retarget --task whisking
```

## Backends

Currently only the **mink** backend (same as do-as-i-do) is wired in. A second
backend based on [dex-retargeting](https://github.com/dexsuite/dex-retargeting)
(`pip install dex_retargeting`, pinocchio+nlopt vector optimization) can be
added; note that its PyPI wheel declares `requires-python <3.13`, so this
package pins `>=3.12,<3.14` to stay compatible with both paths.
