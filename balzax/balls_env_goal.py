import jax
import jax.numpy as jnp

from balzax.balls_base import BallsBase
from balzax.env import GoalEnvState, BalzaxGoalEnv
    

def compute_goal_l2_dist_2(goal_a: jnp.ndarray, goal_b: jnp.ndarray):
    """Returns L2 distance at square between observation and desired goal."""
    return - jnp.sum((goal_a - goal_b)**2)

def compute_goal_l2_dist(goal_a: jnp.ndarray, goal_b: jnp.ndarray):
    """Returns L2 distance between observation and desired goal."""
    return - jnp.linalg.norm(goal_a - goal_b)

def compute_similarity(image_a: jnp.ndarray, image_b: jnp.ndarray):
    """Returns a similarity measure between two sets (image observations)"""
    a_bool = jnp.array(image_a, dtype=jnp.bool_)
    b_bool = jnp.array(image_b, dtype=jnp.bool_)
    inter = jnp.sum(a_bool & b_bool)
    union = jnp.sum(a_bool | b_bool)
    return inter / union


class BallsEnvGoal(BalzaxGoalEnv, BallsBase):
    """Balls RL environment with goal specification"""
    
    def __init__(self, 
                 obs_type : str = 'position', 
                 goal_projection : str = 'identity',
                 max_timestep : int = 10000):
        BallsBase.__init__(self, obs_type=obs_type)
        
        self.goal_reward_fcts = {
                            'position': compute_goal_l2_dist_2, 
                            'image': compute_similarity
                            }
        self.compute_goal_reward = self.goal_reward_fcts.get(self.obs_type)
        
        self.goal_projections = {
            'identity': lambda obs: obs
            }
        self.compute_goal_projection = self.goal_projections[goal_projection]
        
        self.max_timestep = jnp.array(max_timestep, dtype=jnp.int32)
    
    def compute_projection(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Computes observation projection on goal space"""
        return self.compute_goal_projection(observation)
    
    def compute_reward(self, 
                       achieved_goal: jnp.ndarray, 
                       desired_goal: jnp.ndarray) -> jnp.ndarray:
        """Computes the reward"""
        return self.compute_goal_reward(achieved_goal, desired_goal)
    
    def set_desired_goal(self, 
                         goal_env_state: GoalEnvState, 
                         desired_goal: jnp.ndarray) -> GoalEnvState:
        """Sets desired goal"""
        new_goalobs = goal_env_state.goalobs
        new_goalobs['desired_goal'] = desired_goal
        return GoalEnvState(key=goal_env_state.key,
                            timestep=goal_env_state.timestep,
                            reward=goal_env_state.reward,
                            done=goal_env_state.done,
                            goalobs=new_goalobs,
                            game_state=goal_env_state.game_state)
    
    def reset_goal(self, key):
        """Picks an initial desired goal randomly"""
        goal_balls, new_key = BallsBase.reset_base(self, key)
        desired_goal = self.get_obs(goal_balls)
        desired_goal = self.compute_projection(desired_goal)
        return new_key, desired_goal
    
    def reset_game(self, key):
        """Picks an initial observation randomly"""
        new_balls, new_key = BallsBase.reset_base(self, key)
        observation = self.get_obs(new_balls)
        achieved_goal = self.compute_projection(observation)
        return new_key, new_balls, observation, achieved_goal
    
    def reset(self, key) -> GoalEnvState:
        """Resets environment and asign a new goal"""
        inter_key, desired_goal = self.reset_goal(key)
        
        new_key, new_balls, observation, achieved_goal = self.reset_game(inter_key)
        
        goalobs = {'observation': observation,
                   'achieved_goal': achieved_goal,
                   'desired_goal': desired_goal}
        
        return GoalEnvState(key=new_key,
                            timestep=jnp.array(0, dtype=jnp.int32),
                            reward=jnp.array(0.),
                            done=jnp.array(False, dtype=jnp.bool_),
                            goalobs=goalobs,
                            game_state=new_balls)
    
    def reset_done(self, goal_env_state: GoalEnvState) -> GoalEnvState:
        """Resets the environment when done."""
        return jax.lax.cond(goal_env_state.done,
                            self.reset,
                            lambda key: goal_env_state,
                            goal_env_state.key)
    
    def step(self, 
             goal_env_state: GoalEnvState, 
             action: jnp.ndarray) -> GoalEnvState:
        """Performs a goal environment step."""
        new_balls = BallsBase.step_base(self, 
                                        goal_env_state.game_state, 
                                        action)
        
        new_observation = self.get_obs(new_balls)
        new_achieved_goal = self.compute_projection(new_observation)
        desired_goal = goal_env_state.goalobs.get('desired_goal')
        new_goalobs = {'observation': new_observation,
                       'achieved_goal': new_achieved_goal,
                       'desired_goal': desired_goal}
        
        reward = self.compute_reward(new_achieved_goal, desired_goal)
        new_timestep = goal_env_state.timestep + 1
        done = BallsBase.done_base(self, new_balls, new_timestep, self.max_timestep)
        
        return GoalEnvState(key=goal_env_state.key,
                            timestep=new_timestep,
                            reward=reward,
                            done=done,
                            goalobs=new_goalobs,
                            game_state=new_balls)
