import gym
from typing import Optional, Tuple, Callable
import jax.numpy as jnp
from balzax.balls_env_goal import BallsEnvGoal
from balzax.wrapper import GoalGymVecWrapper


def gym_balls_env_goal_factory(
    name: str,
    obs_type: str = "position",
    num_balls: int = 1,
    goal_projection: str = "identity",
    max_episode_steps: int = 500,
    num_envs: int = 1,
    seed: int = 0,
    backend: Optional[str] = None,
):
    gym.register(
        id=name,
        entry_point="balzax.gym_balls_env_goal:GymBallsEnvGoal",
        max_episode_steps=None,
        kwargs={
            "obs_type": obs_type,
            "num_balls": num_balls,
            "goal_projection": goal_projection,
            "max_episode_steps": max_episode_steps,
            "num_envs": num_envs,
            "seed": seed,
            "backend": backend,
        },
    )


class GymBallsEnvGoal(GoalGymVecWrapper):
    """Gym environment wrapping a vectorized BallsEnvGoal environment"""

    def __init__(
        self,
        obs_type: str = "position",
        num_balls: int = 1,
        goal_projection: str = "identity",
        max_episode_steps: int = 500,
        num_envs: int = 1,
        seed: int = 0,
        backend: Optional[str] = None,
        projection_fct: Tuple[
            str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ] = None,
        reward_fct: Tuple[
            str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ] = None,
        success_fct: Tuple[
            str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        ] = None,
    ):
        env = BallsEnvGoal(
            obs_type=obs_type,
            num_balls=num_balls,
            goal_projection=goal_projection,
            max_episode_steps=max_episode_steps,
        )

        if projection_fct is not None:
            keyword, function = projection_fct
            env.add_set_goal_projection(keyword, function)

        if reward_fct is not None:
            keyword, function = reward_fct
            env.add_set_goal_reward_fct(keyword, function)

        if success_fct is not None:
            keyword, function = success_fct
            env.add_set_goal_is_success(keyword, function)

        super().__init__(env=env, num_envs=num_envs, seed=seed, backend=backend)
        self.max_episode_steps = max_episode_steps
