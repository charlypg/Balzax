import os
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
from balzax.balls.gym_balls_env_goal import gym_balls_env_goal_factory
from xpag.agents import SAC
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.setters import DefaultSetter
from xpag.tools.learn import learn
from xpag.wrappers import gym_vec_env
from plotting import single_rollout_eval_custom

# Verifications
# Arguments
assert len(sys.argv) == 3
config_file = sys.argv[1]
assert type(config_file) == str
num_balls_str = sys.argv[2]
assert num_balls_str in {"1", "2"}

# Backend
assert jax.lib.xla_bridge.get_backend().platform == "gpu"

# Functions for goal-oriented environment and plotting


def projection_function_2(obs: jnp.ndarray):
    """Projection into the goal space"""
    return obs[2:]


if num_balls_str == "2":
    projection_fct = ("custom_proj", projection_function_2)
else:
    projection_fct = None


def sparse_reward(goal_a: jnp.ndarray, goal_b: jnp.ndarray):
    return jnp.array([jnp.sum((goal_a - goal_b) ** 2) <= (0.03**2)]) * 1.0


reward_fct = ("sparse_reward", sparse_reward)


def success_function(goal_a: jnp.ndarray, goal_b: jnp.ndarray):
    return jnp.array([jnp.sum((goal_a - goal_b) ** 2) <= (0.03**2)])


success_fct = ("custom_success", success_function)


def plot_proj(x: np.ndarray):
    """Projection for plots"""
    return x.reshape((x.shape[0] // 2, 2))


plot_projection = plot_proj

# Hyperparameters
# Loading hyperparameters
file = open(config_file, "r")
json_str = file.read()
file.close()
hyperparameters = json.loads(json_str)

# Agent
HIDDEN_DIMS = tuple(hyperparameters[num_balls_str]["agent_params"]["HIDDEN_DIMS"])
ACTOR_LR = hyperparameters[num_balls_str]["agent_params"]["ACTOR_LR"]
CRITIC_LR = hyperparameters[num_balls_str]["agent_params"]["CRITIC_LR"]
TEMP_LR = hyperparameters[num_balls_str]["agent_params"]["TEMP_LR"]
TAU = hyperparameters[num_balls_str]["agent_params"]["TAU"]
SEED_AGENT = hyperparameters[num_balls_str]["agent_params"]["SEED_AGENT"]

# TRAINING
NUM_ENVS = hyperparameters[num_balls_str]["training_params"]["NUM_ENVS"]
RATIO_SAVE_STEP_MAX_STEP = hyperparameters[num_balls_str]["training_params"][
    "RATIO_SAVE_STEP_MAX_STEP"
]
BATCH_SIZE = hyperparameters[num_balls_str]["training_params"]["BATCH_SIZE"]
BUFFER_SIZE = hyperparameters[num_balls_str]["training_params"]["BUFFER_SIZE"]
GD_STEPS_PER_STEP = hyperparameters[num_balls_str]["training_params"][
    "GD_STEPS_PER_STEP"
]
MAX_STEPS = hyperparameters[num_balls_str]["training_params"]["MAX_STEPS"]
EVALUATE_EVERY_X_STEPS = hyperparameters[num_balls_str]["training_params"][
    "EVALUATE_EVERY_X_STEPS"
]
SAVE_AGENT_EVERY_X_STEPS = hyperparameters[num_balls_str]["training_params"][
    "SAVE_AGENT_EVERY_X_STEPS"
]
SAVE_DIR = os.path.join(
    os.path.curdir,
    "results",
    "xpag",
    "sac_train_balzax",
)
SAVE_EPISODE = hyperparameters[num_balls_str]["training_params"]["SAVE_EPISODE"]
SEED = hyperparameters[num_balls_str]["training_params"]["SEED"]

# Define envs
ENV_NAME = "test_balls"
gym_balls_env_goal_factory(
    name=ENV_NAME,
    obs_type="position",
    max_episode_steps=300,
    num_balls=int(num_balls_str),
    backend="gpu",
    projection_fct=projection_fct,
    reward_fct=reward_fct,
    success_fct=success_fct,
)

env, eval_env, env_info = gym_vec_env(ENV_NAME, NUM_ENVS)
print(env_info)

# Define agent, sampler, buffer, setter
agent = SAC(
    env_info["observation_dim"]
    if not env_info["is_goalenv"]
    else env_info["observation_dim"] + env_info["desired_goal_dim"],
    env_info["action_dim"],
    {
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "temp_lr": TEMP_LR,
        "tau": TAU,
        "seed": SEED_AGENT,
    },
)
sampler = (
    DefaultEpisodicSampler() if not env_info["is_goalenv"] else HER(env.compute_reward)
)
buffer = DefaultEpisodicBuffer(
    max_episode_steps=env_info["max_episode_steps"],
    buffer_size=BUFFER_SIZE,
    sampler=sampler,
)
setter = DefaultSetter()

# Learning loop
start_training_after_x_steps = RATIO_SAVE_STEP_MAX_STEP * env_info["max_episode_steps"]

learn(
    env,
    eval_env,
    env_info,
    agent,
    buffer,
    setter,
    batch_size=BATCH_SIZE,
    gd_steps_per_step=GD_STEPS_PER_STEP,
    start_training_after_x_steps=start_training_after_x_steps,
    max_steps=MAX_STEPS,
    evaluate_every_x_steps=EVALUATE_EVERY_X_STEPS,
    save_agent_every_x_steps=SAVE_AGENT_EVERY_X_STEPS,
    save_dir=SAVE_DIR,
    save_episode=SAVE_EPISODE,
    plot_projection=plot_projection,
    custom_eval_function=single_rollout_eval_custom,
    additional_step_keys=None,
    seed=SEED,
)
