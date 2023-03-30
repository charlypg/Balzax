import jax
import jax.numpy as jnp

from balzax.env import BalzaxEnv, EnvState
from balzax.balls.structures import Ball
from balzax.balls.balls_base import BallsBase


def reward_bottom_left_corner(balls: Ball):
    """Negative reward corresponding to the distance of balls to bottom left
    corner."""
    return -jnp.array([jnp.sum(balls.pos**2)])


def sparse_reward_bottom_left_corner(balls: Ball):
    """Sparse reward : 1 if bottom left corner reached, 0 else."""
    return jnp.array((jnp.array([jnp.sum(balls.pos**2)]) < 0.05), dtype=jnp.float32)


class BallsEnv(BalzaxEnv, BallsBase):
    """Balls RL environment"""

    def __init__(
        self,
        obs_type: str = "position",
        num_balls: int = 4,
        max_episode_steps: int = 500,
    ):
        BallsBase.__init__(self, obs_type=obs_type, num_balls=num_balls)
        self.max_episode_steps = jnp.array(max_episode_steps, dtype=jnp.int32)

    def render(self, env_state: EnvState):
        """Returns an image of the scene"""
        return BallsBase.get_image(self, env_state.game_state)

    def compute_reward(self, balls: Ball, action: jnp.ndarray, new_balls: Ball):
        """Returns the reward associated to a transition"""
        return sparse_reward_bottom_left_corner(new_balls)

    def reset(self, key) -> EnvState:
        """Resets the environment step"""
        new_balls, new_key = BallsBase.reset_base(self, key)

        terminated = jnp.array([False], dtype=jnp.bool_)
        truncated = jnp.array([False], dtype=jnp.bool_)
        metrics = dict()

        return EnvState(
            key=new_key,
            timestep=jnp.array([0], dtype=jnp.int32),
            reward=jnp.array([0.0]),
            terminated=terminated,
            truncated=truncated,
            obs=self.get_obs(new_balls),
            game_state=new_balls,
            metrics=metrics,
        )

    def reset_done(self, env_state: EnvState, done: jnp.ndarray) -> EnvState:
        """Resets the environment when done."""
        pred = done.squeeze(-1)
        return jax.lax.cond(pred, self.reset, lambda key: env_state, env_state.key)

    def step(self, env_state: EnvState, action: jnp.ndarray) -> EnvState:
        """Performs an environment step."""
        new_balls = BallsBase.step_base(self, env_state.game_state, action)
        new_obs = self.get_obs(new_balls)
        reward = self.compute_reward(env_state.game_state, action, new_balls)
        terminated = jnp.array(reward, dtype=jnp.bool_)
        new_timestep = env_state.timestep + 1
        metrics = dict()
        truncated_b = BallsBase.truncated_base(self, new_balls)
        truncated = new_timestep >= self.max_episode_steps
        truncated = jnp.logical_or(truncated, truncated_b)
        truncated = jnp.logical_and(truncated, jnp.logical_not(terminated))
        return EnvState(
            key=env_state.key,
            timestep=new_timestep,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            obs=new_obs,
            game_state=new_balls,
            metrics=metrics,
        )

    @property
    def observation_low(self):
        return 0.0

    @property
    def observation_high(self):
        return 1.0

    @property
    def action_shape(self):
        return (2,)

    @property
    def action_low(self):
        return -1.0

    @property
    def action_high(self):
        return 1.0
