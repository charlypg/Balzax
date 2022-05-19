import jax
import jax.numpy as jnp
from typing import Callable

from balzax.balls_base import BallsBase
from balzax.env import GoalEnvState, BalzaxGoalEnv


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
        max_episode_steps: int = 500,
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

    def render(self, goal_env_state: GoalEnvState):
        """Returns an image of the scene"""
        return BallsBase.get_image(self, goal_env_state.game_state)

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
        self, goal_env_state: GoalEnvState, desired_goal: jnp.ndarray
    ) -> GoalEnvState:
        """Sets desired goal"""
        new_goalobs = goal_env_state.goalobs
        new_goalobs["desired_goal"] = desired_goal
        return GoalEnvState(
            key=goal_env_state.key,
            timestep=goal_env_state.timestep,
            reward=goal_env_state.reward,
            done=goal_env_state.done,
            goalobs=new_goalobs,
            game_state=goal_env_state.game_state,
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

    def reset(self, key) -> GoalEnvState:
        """Resets environment and asign a new goal"""
        inter_key, desired_goal = self.reset_goal(key)

        new_key, new_balls, observation, achieved_goal = self.reset_game(inter_key)

        goalobs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

        is_success = self.compute_is_success(achieved_goal, desired_goal)
        truncation = jnp.array([False], dtype=jnp.bool_)
        metrics = {"is_success": is_success, "truncation": truncation}

        done = is_success | truncation

        return GoalEnvState(
            key=new_key,
            timestep=jnp.array([0], dtype=jnp.int32),
            reward=jnp.array([0.0]),
            done=done,
            goalobs=goalobs,
            game_state=new_balls,
            metrics=metrics,
        )

    def reset_done(self, goal_env_state: GoalEnvState) -> GoalEnvState:
        """Resets the environment when done."""
        pred = goal_env_state.done.squeeze(-1)
        return jax.lax.cond(
            pred, self.reset, lambda key: goal_env_state, goal_env_state.key
        )

    def step(self, goal_env_state: GoalEnvState, action: jnp.ndarray) -> GoalEnvState:
        """Performs a goal environment step."""
        new_balls = BallsBase.step_base(self, goal_env_state.game_state, action)

        new_observation = self.get_obs(new_balls)
        new_achieved_goal = self.compute_projection(new_observation)
        desired_goal = goal_env_state.goalobs.get("desired_goal")
        new_goalobs = {
            "observation": new_observation,
            "achieved_goal": new_achieved_goal,
            "desired_goal": desired_goal,
        }

        reward = self.compute_reward(new_achieved_goal, desired_goal)
        new_timestep = goal_env_state.timestep + 1
        done_b = BallsBase.done_base(self, new_balls)

        is_success = self.compute_is_success(new_achieved_goal, desired_goal)
        truncation = new_timestep >= self.max_episode_steps
        metrics = {
            "is_success": is_success,
            "truncation": truncation & jnp.logical_not(is_success),
        }

        done = is_success | truncation | done_b

        return GoalEnvState(
            key=goal_env_state.key,
            timestep=new_timestep,
            reward=reward,
            done=done,
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
        return (1,)

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
