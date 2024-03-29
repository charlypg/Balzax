import jax
import jax.numpy as jnp
from typing import Callable

from balzax.env import EnvState, BalzaxGoalEnv
from balzax.balls.balls_base import BallsBase


def compute_goal_l2_dist_2(goal_a: jnp.ndarray, goal_b: jnp.ndarray) -> jnp.ndarray:
    """Returns L2 distance at square between observation and desired goal."""
    return jnp.array([-jnp.sum((goal_a - goal_b) ** 2)])


def compute_goal_l2_dist(goal_a: jnp.ndarray, goal_b: jnp.ndarray) -> jnp.ndarray:
    """Returns L2 distance between observation and desired goal."""
    return jnp.array([-jnp.linalg.norm(goal_a - goal_b)])


def compute_similarity(image_a: jnp.ndarray, image_b: jnp.ndarray) -> jnp.ndarray:
    """Returns a similarity measure between two sets (image observations)"""
    a_bool = jnp.array(image_a, dtype=jnp.bool_)
    b_bool = jnp.array(image_b, dtype=jnp.bool_)
    inter = jnp.sum(a_bool & b_bool)
    union = jnp.sum(a_bool | b_bool)
    return jnp.array([inter / union])


def default_is_success(
    achieved_goal: jnp.ndarray, desired_goal: jnp.ndarray
) -> jnp.ndarray:
    """Computes a boolean indicating whether the goal is reached or not"""
    return jnp.array([False], dtype=jnp.bool_)


class BallsEnvGoal(BalzaxGoalEnv, BallsBase):
    """Balls RL environment with goal specification"""

    def __init__(
        self,
        obs_type: str = "position",
        num_balls: int = 4,
        goal_projection: str = "identity",
        goal_success: str = "default",
        max_episode_steps: int = 300,
    ):
        BallsBase.__init__(self, obs_type=obs_type, num_balls=num_balls)

        self.goal_reward_fcts = dict()
        self.compute_goal_reward = None
        self.add_goal_reward_fct("position", compute_goal_l2_dist_2)
        self.add_goal_reward_fct("image", compute_similarity)
        self.set_goal_reward_fct(obs_type)

        self.goal_projections = dict()
        self.compute_goal_projection = None
        self.add_goal_projection("identity", lambda obs: obs)
        self.set_goal_projection(goal_projection)

        self.goal_is_success = dict()
        self.compute_goal_is_success = None
        self.add_goal_is_success("default", default_is_success)
        self.set_goal_is_success(goal_success)

        self.max_episode_steps = jnp.array(max_episode_steps, dtype=jnp.int32)

    def render(self, env_state: EnvState):
        """Returns an image of the scene"""
        return BallsBase.get_image(self, env_state.game_state)

    def compute_projection(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Computes observation projection on goal space"""
        return self.compute_goal_projection(observation)

    def compute_reward(
        self, achieved_goal: jnp.ndarray, desired_goal: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes the reward"""
        return self.compute_goal_reward(achieved_goal, desired_goal)

    def compute_is_success(self, achieved_goal, desired_goal) -> jnp.ndarray:
        """Computes a boolean indicating whether the goal is reached or not"""
        return self.compute_goal_is_success(achieved_goal, desired_goal)

    def set_desired_goal(
        self, env_state: EnvState, desired_goal: jnp.ndarray
    ) -> EnvState:
        """Sets desired goal"""
        new_obs = env_state.obs.copy()
        new_obs["desired_goal"] = desired_goal
        return EnvState(
            key=env_state.key,
            timestep=env_state.timestep,
            reward=env_state.reward,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            goalobs=new_obs,
            game_state=env_state.game_state,
        )

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

    def reset(self, key) -> EnvState:
        """Resets environment and asign a new goal"""
        inter_key, desired_goal = self.reset_goal(key)

        new_key, new_balls, observation, achieved_goal = self.reset_game(inter_key)

        obs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

        is_success = self.compute_is_success(achieved_goal, desired_goal)
        terminated = is_success.copy()
        reward = self.compute_reward(achieved_goal, desired_goal)
        truncated = jnp.array([False], dtype=jnp.bool_)
        metrics = dict(is_success=is_success.copy())

        return EnvState(
            key=new_key,
            timestep=jnp.array([0], dtype=jnp.int32),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            obs=obs,
            game_state=new_balls,
            metrics=metrics,
        )

    def reset_done(self, env_state: EnvState, done: jnp.ndarray) -> EnvState:
        """Resets the environment when done."""
        pred = done.squeeze(-1)
        return jax.lax.cond(pred, self.reset, lambda key: env_state, env_state.key)

    def step(self, env_state: EnvState, action: jnp.ndarray) -> EnvState:
        """Performs a goal environment step."""
        new_balls = BallsBase.step_base(self, env_state.game_state, action)

        new_observation = self.get_obs(new_balls)
        new_achieved_goal = self.compute_projection(new_observation)
        desired_goal = env_state.goalobs.get("desired_goal")
        new_goalobs = {
            "observation": new_observation,
            "achieved_goal": new_achieved_goal,
            "desired_goal": desired_goal,
        }

        reward = self.compute_reward(new_achieved_goal, desired_goal)
        new_timestep = env_state.timestep + 1
        truncated_b = BallsBase.truncated_base(self, new_balls)

        is_success = self.compute_is_success(new_achieved_goal, desired_goal)
        terminated = is_success.copy()
        truncated = new_timestep >= self.max_episode_steps
        metrics = dict(is_success=is_success.copy())

        truncated = jnp.logical_or(truncated, truncated_b)
        truncated = jnp.logical_and(truncated, jnp.logical_not(terminated))

        return EnvState(
            key=env_state.key,
            timestep=new_timestep,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            goalobs=new_goalobs,
            game_state=new_balls,
            metrics=metrics,
        )

    @property
    def goal_low(self):
        return 0.0

    @property
    def goal_high(self):
        return 1.0

    @property
    def observation_low(self):
        return 0.0

    @property
    def observation_high(self):
        return 1.0

    @property
    def action_shape(self):
        return (2,)

    @property
    def action_low(self):
        return -1.0

    @property
    def action_high(self):
        return 1.0

    def add_goal_reward_fct(
        self, keyword: str, function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ):
        self.goal_reward_fcts[keyword] = function

    def set_goal_reward_fct(self, keyword: str):
        self.compute_goal_reward = self.goal_reward_fcts.get(keyword)

    def add_set_goal_reward_fct(
        self, keyword: str, function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ):
        self.add_goal_reward_fct(keyword, function)
        self.set_goal_reward_fct(keyword)

    def add_goal_projection(
        self, keyword: str, function: Callable[[jnp.ndarray], jnp.ndarray]
    ):
        self.goal_projections[keyword] = function

    def set_goal_projection(self, keyword: str):
        self.compute_goal_projection = self.goal_projections.get(keyword)

    def add_set_goal_projection(
        self, keyword: str, function: Callable[[jnp.ndarray], jnp.ndarray]
    ):
        self.add_goal_projection(keyword, function)
        self.set_goal_projection(keyword)

    def add_goal_is_success(
        self, keyword: str, function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ):
        self.goal_is_success[keyword] = function

    def set_goal_is_success(self, keyword: str):
        self.compute_goal_is_success = self.goal_is_success.get(keyword)

    def add_set_goal_is_success(
        self, keyword: str, function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ):
        self.add_goal_is_success(keyword, function)
        self.set_goal_is_success(keyword)
