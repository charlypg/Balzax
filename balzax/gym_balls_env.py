from typing import Optional
from balzax.balls_env import BallsEnv
from balzax.wrapper import GymWrapper, GymVecWrapper


class GymSingleBallsEnv(GymWrapper):
    """Gym environment wrapping a BallsEnv environment"""

    def __init__(
        self,
        obs_type: str = "position",
        num_balls: int = 4,
        max_timestep: int = 10000,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        env = BallsEnv(
            obs_type=obs_type, num_balls=num_balls, max_timestep=max_timestep
        )
        super().__init__(env=env, seed=seed, backend=backend)
        self.max_episode_steps = max_timestep


class GymBallsEnv(GymVecWrapper):
    """Gym environment wrapping a BallsEnv environment"""

    def __init__(
        self,
        obs_type: str = "position",
        num_balls: int = 4,
        max_timestep: int = 10000,
        num_envs: int = 1,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        env = BallsEnv(
            obs_type=obs_type, num_balls=num_balls, max_timestep=max_timestep
        )
        super().__init__(env=env, num_envs=num_envs, seed=seed, backend=backend)
        self.max_episode_steps = max_timestep
