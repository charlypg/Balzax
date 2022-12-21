import jax
import jax.numpy as jnp
import numpy as onp
import gym
from gym import error
from abc import abstractmethod
from typing import Optional, Dict
from balzax.env import BalzaxEnv, BalzaxGoalEnv


def jnpdict_to_onpdict(jnp_dict: Dict[str, jnp.ndarray]):
    """Converts a dictionary with jax.numpy values to one with numpy values"""
    onp_dict = dict()
    for key, value in zip(jnp_dict.keys(), jnp_dict.values()):
        onp_dict[key] = onp.array(value)
    return onp_dict

# TODO: adapt reset(_done) to new Gym interface
class GymWrapper(gym.Env):
    """A wrapper that converts Balzax Env to one that follows Gym API."""

    def __init__(self, env: BalzaxEnv, seed: int = 0, backend: Optional[str] = None):
        self.env = env
        self.key = None
        self.seed(seed)
        self.backend = backend
        self.env_state = None

        # Observation space
        obs_high = self.env.observation_high * onp.ones(
            self.env.observation_shape, dtype=onp.float32
        )
        obs_low = self.env.observation_low * onp.ones(
            self.env.observation_shape, dtype=onp.float32
        )
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype="float32")

        # Action space
        action_high = self.env.action_high * onp.ones(
            self.env.action_shape, dtype=onp.float32
        )
        action_low = self.env.action_low * onp.ones(
            self.env.action_shape, dtype=onp.float32
        )
        self.action_space = gym.spaces.Box(action_low, action_high, dtype="float32")

        # jit functions : BalzaxEnv dynamics
        self.reset_be = jax.jit(self.env.reset, backend=self.backend)
        self.reset_done_be = jax.jit(self.env.reset_done, backend=self.backend)
        self.step_be = jax.jit(self.env.step, backend=self.backend)
        self.render_be = jax.jit(self.env.render, backend=self.backend)

    def seed(self, seed: int = 0):
        """Seed for pseudo-random number generation"""
        self.key = jax.random.PRNGKey(seed)

    def render(self, mode="image"):
        """Rendering of the game state : image by default"""
        return self.render_be(self.env_state)

    def reset(self, return_info: bool = False):
        """Resets env state"""
        self.env_state = self.reset_be(self.key)
        self.key = self.env_state.key
        if return_info:
            info = self.env_state.metrics.copy()
            info.update(self.env_state.info)
            return self.env_state.obs, info
        else:
            return self.env_state.obs

    def reset_done(self, return_info: bool = False):
        """Resets env when done is true"""
        self.env_state = self.reset_done_be(self.env_state)
        if return_info:
            info = self.env_state.metrics.copy()
            info.update(self.env_state.info)
            return self.env_state.obs, info
        else:
            return self.env_state.obs

    def step(self, action):
        """Performs an env step"""
        self.env_state = self.step_be(self.env_state, action)
        return (
            self.env_state.obs,
            self.env_state.reward,
            self.env_state.terminated,
            self.env_state.truncated,
            self.env_state.metrics,
        )

# TODO: adapt reset(_done) to new Gym interface
class GymWrapperSB3(GymWrapper):
    """Gym wrapper which can be used with stable-baselines3"""

    def render(self, mode="image"):
        """Rendering of the game state : image by default"""
        return onp.array(super().render(mode=mode))

    def reset(self, return_info: bool = False):
        """Resets env state"""
        return onp.array(super().reset(return_info=return_info))

    def reset_done(self, return_info: bool = False):
        """Resets env when done is true"""
        return onp.array(super().reset_done(return_info=return_info))

    def step(self, action: onp.ndarray):
        """Performs an env step"""
        obs, reward, terminated, truncated, info = super().step(jnp.array(action))
        obs = onp.array(obs)
        reward = onp.array(reward.squeeze(-1))
        terminated = onp.array(terminated.squeeze(-1))
        truncated = onp.array(truncated.squeeze(-1))
        info = jnpdict_to_onpdict(info)
        return obs, reward, terminated, truncated, info

# TODO: adapt reset(_done) to new Gym interface
class GymVecWrapper(gym.Env):
    """Vectorized version of GymWrapper.
    This wrapper that converts a vectorized Balzax Env to a Gym Env."""

    def __init__(
        self,
        env: BalzaxEnv,
        num_envs: int = 1,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.keys = None
        self.seed(seed)
        self.backend = backend
        self.env_state = None

        # Observation space
        obs_shape = self.env.observation_shape

        obs_high = self.env.observation_high * onp.ones(obs_shape, dtype=onp.float32)
        obs_low = self.env.observation_low * onp.ones(obs_shape, dtype=onp.float32)
        self.single_observation_space = gym.spaces.Box(
            obs_low, obs_high, dtype="float32"
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

        # Action space
        action_high = self.env.action_high * onp.ones(
            self.env.action_shape, dtype=onp.float32
        )
        action_low = self.env.action_low * onp.ones(
            self.env.action_shape, dtype=onp.float32
        )
        self.single_action_space = gym.spaces.Box(
            action_low, action_high, dtype="float32"
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )

        # jit functions : BalzaxEnv dynamics
        self.reset_be = jax.jit(jax.vmap(self.env.reset), backend=self.backend)
        self.reset_done_be = jax.jit(
            jax.vmap(self.env.reset_done), backend=self.backend
        )
        self.step_be = jax.jit(jax.vmap(self.env.step), backend=self.backend)
        self.render_be = jax.jit(jax.vmap(self.env.render), backend=self.backend)

    def seed(self, seed: int = 0):
        """Seed for pseudo-random number generation"""
        key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(key, num=self.num_envs)

    def render(self, mode="image"):
        """Rendering of the game state : image by default"""
        return self.render_be(self.env_state)

    def reset(self, return_info: bool = False):
        """Resets env state"""
        self.env_state = self.reset_be(self.keys)
        self.keys = self.env_state.key
        if return_info:
            info = self.env_state.metrics.copy()
            info.update(self.env_state.info)
            return self.env_state.obs, info
        else:
            return self.env_state.obs

    def reset_done(self, return_info: bool = False):
        """Resets env when done is true"""
        self.env_state = self.reset_done_be(self.env_state)
        if return_info:
            info = self.env_state.metrics.copy()
            info.update(self.env_state.info)
            return self.env_state.obs, info
        else:
            return self.env_state.obs

    def step(self, action):
        """Performs an env step"""
        self.env_state = self.step_be(self.env_state, action)
        return (
            self.env_state.obs,
            self.env_state.reward,
            self.env_state.terminated,
            self.env_state.truncated,
            self.env_state.metrics,
        )


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


class GoalGymVecWrapper(GoalEnv):
    """Vectorized version of GoalEnv.
    This wrapper that converts a vectorized BalzaxGoalEnv to a Gym Env."""

    def __init__(
        self,
        env: BalzaxGoalEnv,
        num_envs: int = 1,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.keys = None
        self.seed(seed)
        self.backend = backend
        self.env_state = None

        # Observation space

        obs_shape, goal_shape = self.env.goalobs_shapes

        obs_high = self.env.observation_high * onp.ones(obs_shape, dtype=onp.float32)
        obs_low = self.env.observation_low * onp.ones(obs_shape, dtype=onp.float32)

        goal_high = self.env.goal_high * onp.ones(goal_shape, dtype=onp.float32)
        goal_low = self.env.goal_low * onp.ones(goal_shape, dtype=onp.float32)

        self.single_observation_space = gym.spaces.Dict(
            dict(
                observation=gym.spaces.Box(obs_low, obs_high, dtype=onp.float32),
                achieved_goal=gym.spaces.Box(goal_low, goal_high, dtype=onp.float32),
                desired_goal=gym.spaces.Box(goal_low, goal_high, dtype=onp.float32),
            )
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

        # Action space
        action_high = self.env.action_high * onp.ones(
            self.env.action_shape, dtype=onp.float32
        )
        action_low = self.env.action_low * onp.ones(
            self.env.action_shape, dtype=onp.float32
        )
        self.single_action_space = gym.spaces.Box(
            action_low, action_high, dtype="float32"
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )

        # jit functions : BalzaxEnv dynamics
        self.reset_be = jax.jit(jax.vmap(self.env.reset), backend=self.backend)
        self.reset_done_be = jax.jit(
            jax.vmap(self.env.reset_done), backend=self.backend
        )
        self.step_be = jax.jit(jax.vmap(self.env.step), backend=self.backend)
        self.render_be = jax.jit(jax.vmap(self.env.render), backend=self.backend)

        self.compute_reward_be = jax.jit(
            jax.vmap(self.env.compute_reward), backend=self.backend
        )

        self.set_desired_goal_be = jax.jit(
            jax.vmap(self.env.set_desired_goal), backend=self.backend
        )

    def seed(self, seed: int = 0):
        """Seed for pseudo-random number generation"""
        key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(key, num=self.num_envs)

    def render(self, mode="image"):
        """Rendering of the game state : image by default"""
        return self.render_be(self.env_state)

    def compute_reward(self, achieved_goal, desired_goal, info=dict()):
        """Computes goal env reward"""
        return self.compute_reward_be(jnp.array(achieved_goal), jnp.array(desired_goal))

    def set_desired_goal(self, goal):
        """Set the goal"""
        self.env_state = self.set_desired_goal_be(self.env_state, jnp.array(goal))

    def reset(
        self, 
        return_info: bool = True,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Resets env state"""
        if seed is not None:
            self.seed(seed=seed)
        self.env_state = self.reset_be(self.keys)
        self.keys = self.env_state.key
        if return_info:
            info = self.env_state.metrics.copy()
            info.update(self.env_state.info)
            return self.env_state.goalobs, info
        else:
            return self.env_state.goalobs

    def reset_done(
        self, 
        done: bool,
        return_info: bool = True,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Resets env when done is true"""
        if seed is not None:
            self.seed(seed=seed)
        self.env_state = self.reset_done_be(self.env_state, done)
        if return_info:
            info = self.env_state.metrics.copy()
            info.update(self.env_state.info)
            return self.env_state.goalobs, info
        else:
            return self.env_state.goalobs

    def step(self, action: jnp.ndarray):
        """Performs an env step"""
        self.env_state = self.step_be(self.env_state, action)
        return (
            self.env_state.goalobs,
            self.env_state.reward,
            self.env_state.terminated,
            self.env_state.truncated,
            self.env_state.metrics,
        )
