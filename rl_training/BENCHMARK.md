# Retargeted-trajectory RL benchmark

The current optimization target is a policy conditioned on both:

- the retargeted dexterous-hand trajectory; and
- the manipulated-object SE(3) trajectory.

Do not compare runs by training reward alone. The primary metric is unassisted,
frame-0 evaluation success.

## Fixed protocol

For each checkpoint, evaluate seeds `42`, `43`, and `44`, at least 500 completed
episodes per seed, with the play configuration (object assistance disabled and
all episodes starting from reference frame zero).

Record:

- success rate;
- object position and orientation error;
- hand joint RMSE;
- wrist-object relative position and orientation error;
- fingertip tracking score;
- contact match, missed contact and unintended contact;
- mean episode length and mean step reward.

The aggregate report should include mean, standard deviation and worst-seed
success rate. A change is accepted only when frame-0 success improves without a
material regression in object tracking or numerical stability.

## Current yellow-spoon baseline

Historical unassisted evaluation artifacts before the latest curriculum/reward
changes report:

| run | checkpoint | episodes | success rate | mean step reward |
|---|---:|---:|---:|---:|
| `rl_training_8192` | 799 | 1024 | 0.0 | 1.8908 |
| `rl_training_8192_v2` | 999 | 1 | 0.0 | 2.8492 |

The second result is not statistically meaningful. A previous 5,859-iteration
training run also showed policy standard deviation growing above 120, while the
environment clipped actions internally. This made the PPO likelihood model
inconsistent with executed actions and caused entropy runaway. The current
configuration clips actions in `RslRlVecEnvWrapper`, uses scalar standard
deviation, and substantially lowers entropy regularization.

## Iteration sequence

1. Two-environment, one-iteration simulator startup smoke test. Also run at least
   256 environments for 10 iterations; the imported floating-root model can show
   extreme state outliers, so reward/observation metrics must remain finite and
   the larger smoke test is the numerical gate.
2. Short 1,000-iteration run; reject immediately for NaN, exploding action std,
   or falling short-episode metrics.
3. Evaluate checkpoints at 200-step intervals with the fixed three-seed protocol.
4. Continue only the best configuration to a long run.
5. Add domain randomization after the non-randomized frame-0 baseline succeeds.

## Current curriculum

Training uses reference-state initialization to learn local tracking, with a
small frame-0 population and reduced reset noise. After sustained local success,
an adaptive curriculum raises the frame-0 probability, increases perturbations,
and removes object assistance. Assistance is competence-gated rather than
wall-clock-decayed, and is always disabled in evaluation.
