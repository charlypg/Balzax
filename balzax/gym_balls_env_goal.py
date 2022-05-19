from typing import Optional
from balzax.balls_env_goal import BallsEnvGoal
from balzax.wrapper import GoalGymVecWrapper


class GymBallsEnvGoal(GoalGymVecWrapper):
    """Gym environment wrapping a vectorized BallsEnvGoal environment"""

    def __init__(
        self,
        obs_type: str = "position",
        num_balls: int = 4,
        goal_projection: str = "identity",
        max_episode_steps: int = 500,
        num_envs: int = 1,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        env = BallsEnvGoal(
            obs_type=obs_type,
            num_balls=num_balls,
            goal_projection=goal_projection,
            max_episode_steps=max_episode_steps,
        )
        super().__init__(env=env, num_envs=num_envs, seed=seed, backend=backend)
        self.max_episode_steps = max_episode_steps
