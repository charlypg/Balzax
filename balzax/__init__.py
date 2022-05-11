import gym

gym.register(
    id="GymSingleBallsEnv-position-v0",
    entry_point="balzax.gym_balls_env:GymSingleBallsEnv",
    max_episode_steps=None,
    kwargs={"obs_type": "position", "backend": "gpu"},
)

gym.register(
    id="GymSingleBallsEnv-image-v0",
    entry_point="balzax.gym_balls_env:GymSingleBallsEnv",
    max_episode_steps=None,
    kwargs={"obs_type": "image", "backend": "gpu"},
)

gym.register(
    id="GymBallsEnv-position-v0",
    entry_point="balzax.gym_balls_env:GymBallsEnv",
    max_episode_steps=None,
    kwargs={"obs_type": "position", "backend": "gpu"},
)

gym.register(
    id="GymBallsEnv-image-v0",
    entry_point="balzax.gym_balls_env:GymBallsEnv",
    max_episode_steps=None,
    kwargs={"obs_type": "image", "backend": "gpu"},
)

gym.register(
    id="GymBallsEnvGoal-position-v0",
    entry_point="balzax.gym_balls_env_goal:GymBallsEnvGoal",
    max_episode_steps=None,
    kwargs={"obs_type": "position", "backend": "gpu"},
)

gym.register(
    id="GymBallsEnvGoal-image-v0",
    entry_point="balzax.gym_balls_env_goal:GymBallsEnvGoal",
    max_episode_steps=None,
    kwargs={"obs_type": "image", "backend": "gpu"},
)

gym.register(
    id="GymBallsEnvGoal-oneball-position-v0",
    entry_point="balzax.gym_balls_env_goal:GymBallsEnvGoal",
    max_episode_steps=None,
    kwargs={"obs_type": "position", "num_balls": 1, "backend": "gpu"},
)

gym.register(
    id="GymBallsEnvGoal-oneball-image-v0",
    entry_point="balzax.gym_balls_env_goal:GymBallsEnvGoal",
    max_episode_steps=None,
    kwargs={"obs_type": "image", "num_balls": 1, "backend": "gpu"},
)
