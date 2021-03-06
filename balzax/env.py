import abc
from typing import Dict, Any
import jax
import jax.numpy as jnp
import flax


@flax.struct.dataclass
class EnvState:
    """Defines the environment state (without goal)"""

    key: jnp.ndarray
    timestep: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    game_state: flax.struct.dataclass
    metrics: Dict[str, jnp.ndarray] = flax.struct.field(default_factory=dict)
    info: Dict[str, Any] = flax.struct.field(default_factory=dict)


class BalzaxEnv(abc.ABC):
    """Defines a Balzax environment without goal"""

    @abc.abstractmethod
    def reset(self, key: jnp.ndarray) -> EnvState:
        """Resets the environment to an initial state"""

    @abc.abstractmethod
    def step(self, env_state: EnvState, action: jnp.ndarray) -> EnvState:
        """Run a timestep of the environment"""

    @abc.abstractmethod
    def reset_done(self, env_state: EnvState) -> EnvState:
        """Resets environment when done"""

    def render(self, env_state: EnvState):
        """Returns a render of the env state"""
        return None

    @property
    def observation_shape(self):
        key = jax.random.PRNGKey(0)
        env_state = self.reset(key)
        return env_state.obs.shape

    @property
    def observation_low(self):
        raise Exception("BalzaxEnv : obs_low Not Implemented in inherited class")

    @property
    def observation_high(self):
        raise Exception("BalzaxEnv : obs_high Not Implemented in inherited class")

    @property
    def action_shape(self):
        raise Exception("BalzaxEnv : action_shape Not Implemented in inherited class")

    @property
    def action_low(self):
        raise Exception("BalzaxEnv : action_low Not Implemented in inherited class")

    @property
    def action_high(self):
        raise Exception("BalzaxEnv : action_high Not Implemented in inherited class")

    @property
    def unwrapped(self):
        return self


@flax.struct.dataclass
class GoalEnvState:
    """Fully describes the system state
    and embeds necessary info for RL
    algorithms + goal specifications"""

    key: jnp.ndarray
    timestep: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    goalobs: Dict[str, jnp.ndarray]
    game_state: flax.struct.dataclass
    metrics: Dict[str, jnp.ndarray] = flax.struct.field(default_factory=dict)
    info: Dict[str, Any] = flax.struct.field(default_factory=dict)


class BalzaxGoalEnv(abc.ABC):
    """Defines a Balzax environment with a goal
    for goal conditioned RL"""

    @abc.abstractmethod
    def compute_projection(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Computes observation projection on goal space"""

    @abc.abstractmethod
    def compute_reward(
        self, achieved_goal: jnp.ndarray, desired_goal: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes the reward"""

    @abc.abstractmethod
    def compute_is_success(
        self, achieved_goal: jnp.ndarray, desired_goal: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes a boolean indicating whether the goal is reached or not"""

    @abc.abstractmethod
    def set_desired_goal(
        self, goal_env_state: GoalEnvState, desired_goal: jnp.ndarray
    ) -> GoalEnvState:
        """Sets desired goal"""

    @abc.abstractmethod
    def reset(self, key: jnp.ndarray) -> GoalEnvState:
        """Resets the environment to an initial state"""

    @abc.abstractmethod
    def step(self, env_state: GoalEnvState, action: jnp.ndarray) -> GoalEnvState:
        """Run a timestep of the environment"""

    @abc.abstractmethod
    def reset_done(self, env_state: GoalEnvState) -> GoalEnvState:
        """Resets environment when done"""

    def render(self, env_state: GoalEnvState):
        """Returns a render of the env state"""
        return None

    @property
    def goalobs_shapes(self):
        key = jax.random.PRNGKey(0)
        env_state = self.reset(key)
        return (
            env_state.goalobs.get("observation").shape,
            env_state.goalobs.get("achieved_goal").shape,
        )

    @property
    def goal_low(self):
        raise Exception("BalzaxEnv : goal_low Not Implemented in inherited class")

    @property
    def goal_high(self):
        raise Exception("BalzaxEnv : goal_high Not Implemented in inherited class")

    @property
    def observation_low(self):
        raise Exception(
            "BalzaxEnv : observation_low Not Implemented in inherited class"
        )

    @property
    def observation_high(self):
        raise Exception(
            "BalzaxEnv : observation_high Not Implemented in inherited class"
        )

    @property
    def action_shape(self):
        raise Exception("BalzaxEnv : action_shape Not Implemented in inherited class")

    @property
    def action_low(self):
        raise Exception("BalzaxEnv : action_low Not Implemented in inherited class")

    @property
    def action_high(self):
        raise Exception("BalzaxEnv : action_high Not Implemented in inherited class")

    @property
    def unwrapped(self):
        return self
