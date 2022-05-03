import jax
import jax.numpy as jnp
import numpy as onp
import gym
from typing import Optional
from balzax.env import BalzaxEnv


class GymWrapper(gym.Env):
    """A wrapper that converts Balzax Env to one that follows Gym API."""
    
    def __init__(self, 
                 env : BalzaxEnv, 
                 seed : int = 0,
                 backend : Optional[str] = None):
        self.env = env
        self.key = None
        self.seed(seed)
        self.backend = backend
        self.env_state = None
        
        # jit functions : BalzaxEnv dynamics
        self.reset_be = jax.jit(self.env.reset, backend=self.backend)
        self.reset_done_be = jax.jit(self.env.reset_done, backend=self.backend)
        self.step_be = jax.jit(self.env.step, backend=self.backend)
        
        # Observation space
        obs_high = self.env.observation_high * onp.ones(self.env.observation_shape, 
                                                        dtype=onp.float32)
        obs_low = self.env.observation_low * onp.ones(self.env.observation_shape, 
                                                      dtype=onp.float32)
        self.observation_space = gym.spaces.Box(obs_low, 
                                                obs_high, 
                                                dtype='float32')
        
        # Action space
        action_high = self.env.action_high * onp.ones(self.env.action_size,
                                                      dtype=onp.float32)
        action_low = self.env.action_low * onp.ones(self.env.action_size,
                                                     dtype=onp.float32)
        self.action_space = gym.spaces.Box(action_low, 
                                           action_high,
                                           dtype='float32')
        
    def seed(self, seed: int = 0):
        self.key = jax.random.PRNGKey(seed)
    
    def reset(self):
        self.env_state = self.reset_be(self.key)
        return onp.array(self.env_state.obs)
    
    def reset_done(self):
        self.env_state = self.reset_done_be(self.env_state)
        return onp.array(self.env_state.obs)
    
    def step(self, action : onp.ndarray):
        self.env_state = self.step_be(self.env_state, jnp.array(action))
        obs = onp.array(self.env_state.obs)
        reward = onp.array(self.env_state.reward)
        done = onp.array(self.env_state.done)
        info = dict()
        return obs, reward, done, info
    
    def render(self, mode='human'):
        return None