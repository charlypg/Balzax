import gym
from typing import Optional
from balzax.balls_env import BallsEnv
from balzax.wrapper import GymWrapper, GymVecWrapper


def gym_single_balls_env_factory(
    name: str,
    obs_type: str = "position",
    num_balls: int = 4,
    max_episode_steps: int = 500,
    seed: int = 0,
    backend: Optional[str] = None,
):
    gym.register(
        id=name,
        entry_point="balzax.gym_balls_env:GymSingleBallsEnv",
        max_episode_steps=None,
        kwargs={
            "obs_type": obs_type,
            "num_balls": num_balls,
            "max_episode_steps": max_episode_steps,
            "seed": seed,
            "backend": backend,
        },
    )


class GymSingleBallsEnv(GymWrapper):
    """Gym environment wrapping a BallsEnv environment"""

    def __init__(
        self,
        obs_type: str = "position",
        num_balls: int = 4,
        max_episode_steps: int = 500,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        env = BallsEnv(
            obs_type=obs_type, num_balls=num_balls, max_episode_steps=max_episode_steps
        )
        super().__init__(env=env, seed=seed, backend=backend)
        self.max_episode_steps = max_episode_steps


def gym_balls_env_factory(
    name: str,
    obs_type: str = "position",
    num_balls: int = 4,
    max_episode_steps: int = 500,
    num_envs: int = 1,
    seed: int = 0,
    backend: Optional[str] = None,
):
    gym.register(
        id=name,
        entry_point="balzax.gym_balls_env:GymBallsEnv",
        max_episode_steps=None,
        kwargs={
            "obs_type": obs_type,
            "num_balls": num_balls,
            "max_episode_steps": max_episode_steps,
            "num_envs": num_envs,
            "seed": seed,
            "backend": backend,
        },
    )


class GymBallsEnv(GymVecWrapper):
    """Gym environment wrapping a BallsEnv environment"""

    def __init__(
        self,
        obs_type: str = "position",
        num_balls: int = 4,
        max_episode_steps: int = 500,
        num_envs: int = 1,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        env = BallsEnv(
            obs_type=obs_type, num_balls=num_balls, max_episode_steps=max_episode_steps
        )
        super().__init__(env=env, num_envs=num_envs, seed=seed, backend=backend)
        self.max_episode_steps = max_episode_steps
