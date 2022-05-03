import jax
import jax.numpy as jnp
import numpy as onp
import gym
from gym import error
from abc import abstractmethod
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


class GymVecWrapper(gym.Env):
    """Vectorized version of GymWrapper.
    This wrapper that converts a vectorized Balzax Env to a Gym Env."""
    
    def __init__(self, 
                 env : BalzaxEnv, 
                 num_envs : int = 1,
                 seed : int = 0,
                 backend : Optional[str] = None):
        self.env = env
        self.num_envs = num_envs
        self.keys = None
        self.seed(seed)
        self.backend = backend
        self.env_state = None
        
        # jit functions : BalzaxEnv dynamics
        self.reset_be = jax.jit(jax.vmap(self.env.reset), 
                                backend=self.backend)
        self.reset_done_be = jax.jit(jax.vmap(self.env.reset_done), 
                                     backend=self.backend)
        self.step_be = jax.jit(jax.vmap(self.env.step), 
                               backend=self.backend)
        
        # Observation space
        obs_high = self.env.observation_high * onp.ones(self.env.observation_shape, 
                                                        dtype=onp.float32)
        obs_low = self.env.observation_low * onp.ones(self.env.observation_shape, 
                                                      dtype=onp.float32)
        self.single_observation_space = gym.spaces.Box(obs_low, 
                                                       obs_high, 
                                                       dtype='float32')
        self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space,
                self.num_envs)
        
        # Action space
        action_high = self.env.action_high * onp.ones(self.env.action_size,
                                                      dtype=onp.float32)
        action_low = self.env.action_low * onp.ones(self.env.action_size,
                                                     dtype=onp.float32)
        self.single_action_space = gym.spaces.Box(action_low, 
                                                  action_high,
                                                  dtype='float32')
        self.action_space = gym.vector.utils.batch_space(
                self.single_action_space,
                self.num_envs)
        
    def seed(self, seed: int = 0):
        key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(key, num=self.num_envs)
    
    def reset(self):
        self.env_state = self.reset_be(self.keys)
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


class GoalEnv(gym.Env):
    """The GoalEnv class that was migrated from gym (v0.22) to gym-robotics.
    We add a set_desired_goal() function."""

    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        super().reset(seed=seed)
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key.'.format(key))

    @abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'],
                                                    ob['desired_goal'], info)
        """
        raise NotImplementedError

    @abstractmethod
    def set_desired_goal(self, goal):
        """Set the goal"""
        raise NotImplementedError