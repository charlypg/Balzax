import jax
import jax.numpy as jnp
import flax

from balzax.structures import Ball
from balzax.balls_base import BallsBase


@flax.struct.dataclass
class EnvState:
    """Fully describes the environment 
    state"""
    key: jnp.ndarray
    timestep: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    obs: jnp.ndarray
    balls: Ball

def reward_center_dists(balls: Ball):
    pos = jnp.array([[0.1, 0.1], [0.22, 0.22], [0.45, 0.45], [0.75, 0.75]])
    return - jnp.sum((balls.pos - pos)**2)


class BallsEnv(BallsBase):
    """Balls RL environment"""
    
    def __init__(self, 
                 obs_type : str = 'position',
                 max_timestep : int = 10000):
        super().__init__(obs_type=obs_type)
        self.max_timestep = jnp.array(max_timestep, dtype=jnp.int32)
    
    def compute_reward(self, 
                       balls: Ball, 
                       action: jnp.ndarray, 
                       new_balls: Ball):
        """Returns the reward associated to a transition"""
        return reward_center_dists(new_balls)
    
    def reset(self, key) -> EnvState:
        """Resets the environment step"""
        new_balls, new_key = BallsBase.reset_base(self, key)
        return EnvState(key=new_key, 
                        timestep=jnp.array(0, dtype=jnp.int32),
                        reward=jnp.array(0.),
                        done=jnp.array(False, dtype=jnp.bool_),
                        obs=self.get_obs(new_balls),
                        balls=new_balls)
    
    def reset_done(self, env_state: EnvState) -> EnvState:
        """Resets the environment when done."""
        return jax.lax.cond(env_state.done,
                            self.reset,
                            lambda key: env_state,
                            env_state.key)
    
    def step(self, env_state: EnvState, action: jnp.ndarray) -> EnvState:
        """Performs an environment step."""
        new_balls = self.step_base(env_state.balls, action)
        new_obs = self.get_obs(new_balls)
        reward = self.compute_reward(env_state.balls, action, new_balls)
        new_timestep = env_state.timestep + 1
        done = self.done_base(new_balls, new_timestep, self.max_timestep)
        return EnvState(key=env_state.key, 
                        timestep=new_timestep,
                        reward=reward,
                        done=done,
                        obs=new_obs,
                        balls=new_balls)
