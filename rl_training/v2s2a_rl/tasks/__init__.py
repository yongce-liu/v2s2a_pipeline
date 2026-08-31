"""Register v2s2a Isaac Lab environments."""

import gymnasium as gym

gym.register(
    id="V2S2A-Trajectory-Hand-v0",
    entry_point="v2s2a_rl.tasks.trajectory_env:TrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "v2s2a_rl.tasks.trajectory_env_cfg:TrajectoryTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": "v2s2a_rl.tasks.agents.rsl_rl_ppo_cfg:V2S2APPORunnerCfg",
    },
)

gym.register(
    id="V2S2A-Trajectory-Hand-Play-v0",
    entry_point="v2s2a_rl.tasks.trajectory_env:TrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "v2s2a_rl.tasks.trajectory_env_cfg:TrajectoryTrackingPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "v2s2a_rl.tasks.agents.rsl_rl_ppo_cfg:V2S2APPORunnerCfg",
    },
)
