import jax.numpy as jnp
import flax

from balzax.structures import State
from balzax.balls_base import BallsBase


@flax.struct.dataclass
class GoalEnvState:
    """Fully describes the system state 
    and embeds necessary info for RL 
    algorithms + goal specifications"""
    key: jnp.ndarray
    timestep: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    state: State
    goal: State
    

def compute_goal_l2_dist_sq(state: State, goal: State):
    """Returns L2 distance at square between observation and desired goal."""
    return - jnp.sum((state.obs - goal.obs)**2)

def compute_goal_l2_dist(state: State, goal: State):
    """Returns L2 distance between observation and desired goal."""
    return - jnp.linalg.norm(state.obs - goal.obs)

def compute_similarity(state: State, goal: State):
    """Returns a similarity measure between two sets (image obs)"""
    obs_bool = jnp.array(state.obs, dtype=jnp.bool_)
    goal_bool = jnp.array(goal.obs, dtype=jnp.bool_)
    inter = jnp.sum(obs_bool & goal_bool)
    union = jnp.sum(obs_bool | goal_bool)
    return inter / union

class BallsEnvGoal(BallsBase):
    """Balls RL environment with goal specification"""
    
    def __init__(self, obs_type : str = 'position'):
        super().__init__(obs_type=obs_type)
        self.goal_reward_fcts = {
                            'position': compute_goal_l2_dist_sq, 
                            'image': compute_similarity
                            }
        self.compute_goal_reward = self.goal_reward_fcts.get(self.obs_type)
    
    def reset(self, key) -> GoalEnvState:
        """Resets environment and asign a new goal"""
        goal_balls, inter_key = BallsBase.reset_base(self, key)
        goal = State(ball=goal_balls,
                     obs=self.get_obs(goal_balls))
        new_balls, new_key = BallsBase.reset_base(self, inter_key)
        state = State(ball=new_balls, 
                      obs=self.get_obs(new_balls))
        return GoalEnvState(key=new_key, 
                            timestep=jnp.array(0, dtype=jnp.int32),
                            reward=jnp.array(0.),
                            done=jnp.array(False, dtype=jnp.bool_),
                            state=state, 
                            goal=goal)
    
    def step(self, goal_env_state: GoalEnvState, action: jnp.ndarray):
        """Performs a goal environment step."""
        new_state = self.common_step(goal_env_state.state, action)
        reward = self.compute_goal_reward(new_state, goal_env_state.goal)
        new_timestep = goal_env_state.timestep + 1
        done = self.compute_done(new_state, new_timestep)
        return GoalEnvState(key=goal_env_state.key, 
                            timestep=new_timestep,
                            reward=reward,
                            done=done,
                            state=new_state,
                            goal=goal_env_state.goal)
