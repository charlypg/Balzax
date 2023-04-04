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
        self.common_initialization(
            env=env, num_envs=num_envs, seed=seed, backend=backend
        )
        self.jit_balzax_env_dyn()
        self.update_action_space()
        self.update_observation_space()

    def common_initialization(
        self,
        env: BalzaxEnv,
        num_envs: int = 1,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        """Initializations common to RL and GCRL cases"""
        self.env = env
        self.num_envs = num_envs
        self.keys = None
        self.seed(seed)
        self.backend = backend
        self.env_state = None

    def jit_balzax_env_dyn(self):
        """jit functions : BalzaxEnv dynamics"""
        self.reset_be = jax.jit(jax.vmap(self.env.reset), backend=self.backend)
        self.reset_done_be = jax.jit(
            jax.vmap(self.env.reset_done), backend=self.backend
        )
        self.step_be = jax.jit(jax.vmap(self.env.step), backend=self.backend)
        self.render_be = jax.jit(jax.vmap(self.env.render), backend=self.backend)

    def update_observation_space(self):
        """Computes gym observation space and save it as attribute"""
        obs_shape = self.env.observation_shape

        obs_high = self.env.observation_high * onp.ones(obs_shape, dtype=onp.float32)
        obs_low = self.env.observation_low * onp.ones(obs_shape, dtype=onp.float32)
        self.single_observation_space = gym.spaces.Box(
            obs_low, obs_high, dtype="float32"
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

    def update_action_space(self):
        """Computes gym action space and save it as attribute"""
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

    def seed(self, seed: int = 0):
        """Seed for pseudo-random number generation"""
        key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(key, num=self.num_envs)

    def render(self, mode="image"):
        """Rendering of the game state : image by default"""
        return self.render_be(self.env_state)

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
            return self.env_state.obs, info
        else:
            return self.env_state.obs

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
        self.env_state = self.reset_done_be(
            self.env_state, jnp.array(done, dtype=jnp.bool_)
        )
        if return_info:
            info = self.env_state.metrics.copy()
            info.update(self.env_state.info)
            return self.env_state.obs, info
        else:
            return self.env_state.obs

    def step(self, action: jnp.ndarray):
        """Performs an env step"""
        self.env_state = self.step_be(self.env_state, action)
        info = self.env_state.metrics.copy()
        info.update(self.env_state.info)
        return (
            self.env_state.obs,
            self.env_state.reward,
            self.env_state.terminated,
            self.env_state.truncated,
            info,
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


class GoalGymVecWrapper(GoalEnv, GymVecWrapper):
    """Vectorized version of GoalEnv.
    This wrapper that converts a vectorized BalzaxGoalEnv to a Gym Env."""

    def __init__(
        self,
        env: BalzaxGoalEnv,
        num_envs: int = 1,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        self.common_initialization(
            env=env, num_envs=num_envs, seed=seed, backend=backend
        )
        self.jit_balzax_env_dyn()
        self.jit_balzax_env_goal_utils()
        self.update_action_space()
        self.update_observation_space()

    def jit_balzax_env_goal_utils(self):
        """jit functions specific to BalzaxGoalEnv (GCRL case)"""
        self.compute_reward_be = jax.jit(
            jax.vmap(self.env.compute_reward), backend=self.backend
        )

        self.set_desired_goal_be = jax.jit(
            jax.vmap(self.env.set_desired_goal), backend=self.backend
        )

    def update_observation_space(self):
        """Computes gym observation space and save it as attribute
        in the GCRL case"""
        obs_shape, goal_shape = self.env.observation_shape

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

    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        action=None,
        next_observation=None,
        info=dict(),
    ):
        """Computes goal env reward"""
        return self.compute_reward_be(jnp.array(achieved_goal), jnp.array(desired_goal))

    def set_desired_goal(self, goal):
        """Set the goal"""
        self.env_state = self.set_desired_goal_be(self.env_state, jnp.array(goal))
