import abc
from typing import Union, Dict, Any
import jax
import jax.numpy as jnp
import flax


@flax.struct.dataclass
class EnvState:
    """Fully describes the system state
    and embeds necessary info for RL
    algorithms
    + goal specifications in the GC case"""

    key: jnp.ndarray
    timestep: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    obs: Union[jnp.ndarray, Dict[str, jnp.ndarray]]
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
    def reset_done(self, env_state: EnvState, done: jnp.ndarray) -> EnvState:
        """Resets environment when done"""

    def render(self, env_state: EnvState):
        """Returns a render of the env state"""
        return None

    @property
    def observation_shape(self):
        key = jax.random.PRNGKey(0)
        env_state = self.reset(key)
        if type(env_state.obs) == dict:
            return (
                env_state.obs.get("observation").shape,
                env_state.obs.get("achieved_goal").shape,
            )
        else:
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


class BalzaxGoalEnv(BalzaxEnv):
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
        self, goal_env_state: EnvState, desired_goal: jnp.ndarray
    ) -> EnvState:
        """Sets desired goal"""

    @property
    def goal_low(self):
        raise Exception("BalzaxGoalEnv : goal_low Not Implemented in inherited class")

    @property
    def goal_high(self):
        raise Exception("BalzaxGoalEnv : goal_high Not Implemented in inherited class")

    @property
    def unwrapped(self):
        return self
