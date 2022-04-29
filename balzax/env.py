import abc

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

class Env(abc.ABC):
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



