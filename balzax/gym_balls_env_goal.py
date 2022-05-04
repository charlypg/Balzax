from typing import Optional
from balzax.balls_env_goal import BallsEnvGoal
from balzax.wrapper import GoalGymVecWrapper

class GymBallsEnvGoal(GoalGymVecWrapper):
    """Gym environment wrapping a BallsEnvGoal environment"""
    def __init__(self, 
                 obs_type : str = 'position', 
                 goal_projection : str = 'identity',
                 max_timestep : int = 10000,
                 num_envs : int = 1,
                 seed : int = 0,
                 backend : Optional[str] = None):
        env = BallsEnvGoal(obs_type=obs_type, 
                           goal_projection=goal_projection,
                           max_timestep=max_timestep)
        super().__init__(env=env, 
                         num_envs=num_envs,
                         seed=seed,
                         backend=backend)
        self.max_episode_steps = max_timestep
