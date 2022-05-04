import gym
from balzax.balls_env import BallsEnv
from balzax.balls_env_goal import BallsEnvGoal

gym.register(
    id="GymBallsEnvGoal-position-v0",
    entry_point="balzax.gym_balls_env_goal:GymBallsEnvGoal",
    max_episode_steps=None,
    kwargs={'obs_type': 'position'}
)

gym.register(
    id="GymBallsEnvGoal-image-v0",
    entry_point="balzax.gym_balls_env_goal:GymBallsEnvGoal",
    max_episode_steps=None,
    kwargs={'obs_type': 'image'}
)