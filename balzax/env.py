import abc
from typing import TypedDict
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

class BalzaxEnv(abc.ABC):
    """Defines a Balzax environment without goal"""
    
    @abc.abstractmethod
    def reset(self, key: jnp.ndarray) -> EnvState:
        """Resets the environment to an initial state"""
    
    @abc.abstractmethod
    def step(self, env_state: EnvState, action: jnp.ndarray) -> EnvState:
        """Run a timestep of the environment"""
    
    @abc.abstractmethod
    def reset_done(self, env_state: EnvState, key: jnp.ndarray) -> EnvState:
        """Resets environment when done"""


class GoalObs(TypedDict):
    """Dictionary for observation and goal representation
    like in the Gym robotics API."""
    observation: jnp.ndarray
    achieved_goal: jnp.ndarray
    desired_goal: jnp.ndarray

@flax.struct.dataclass
class GoalEnvState:
    """Fully describes the system state 
    and embeds necessary info for RL 
    algorithms + goal specifications"""
    key: jnp.ndarray
    timestep: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    goalobs: GoalObs
    game_state: flax.struct.dataclass

class BalzaxGoalEnv(abc.ABC):
    """Defines a Balzax environment with a goal
    for goal conditioned RL"""
    
    @abc.abstractmethod
    def reset(self, key: jnp.ndarray) -> GoalEnvState:
        """Resets the environment to an initial state"""
    
    @abc.abstractmethod
    def step(self, env_state: GoalEnvState, action: jnp.ndarray) -> GoalEnvState:
        """Run a timestep of the environment"""
    
    @abc.abstractmethod
    def reset_done(self, env_state: GoalEnvState, key: jnp.ndarray) -> GoalEnvState:
        """Resets environment when done"""
