import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from time import time

from balzax.balls.balls_env_goal import BallsEnvGoal
from balzax.wrapper import GoalGymVecWrapper

# TODO: Animation
# TODO: Test actually used gym environments


@jax.jit
def compute_done(terminated: jnp.ndarray, truncated: jnp.ndarray) -> jnp.ndarray:
    return jnp.logical_or(terminated, truncated)


def plot_vect_goalobs(i: int, vect_goalobs: dict, num_goalobs: int):
    fig = plt.figure(i, constrained_layout=True)
    fig.suptitle("Timestep {}".format(i))

    subfigs = fig.subfigures(nrows=num_goalobs, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle("Environment {}".format(row))

        axs = subfig.subplots(nrows=1, ncols=3)
        for ax, field, images in zip(axs, vect_goalobs.keys(), vect_goalobs.values()):
            ax.imshow(images[row])
            ax.set_title(field)


OBS_TYPE = "image"
NUM_ENVS = 3
SEED = 0
MAX_EPISODE_STEPS = 5
NB_ITER = 21
PULSE = 2 * onp.pi / NB_ITER * onp.ones((NUM_ENVS, 1))

env = BallsEnvGoal(obs_type=OBS_TYPE, max_episode_steps=MAX_EPISODE_STEPS)
gym_env = GoalGymVecWrapper(env=env, num_envs=NUM_ENVS, seed=SEED)

t0 = time()
obs = gym_env.reset(return_info=False)
delta = time() - t0
print("gym_env.reset : {}".format(delta))
obs_list = [obs]
info_list = []

t0 = time()
for i in range(NB_ITER):
    angle = onp.sin(PULSE * i)
    cos = onp.cos(angle)
    sin = onp.sin(angle)
    action = onp.concatenate((cos, sin), axis=1)
    obs, reward, terminated, truncated, info = gym_env.step(action)
    info_list.append(info)
    done = compute_done(terminated, truncated)
    obs = gym_env.reset_done(done, return_info=False)
    obs_list.append(obs)
delta = time() - t0
print("Rollout of {0} : {1}".format(NB_ITER, delta))

"""
num_goalobs = min(2, NUM_ENVS)
for i, vect_goalobs in enumerate(obs_list):
    plot_vect_goalobs(i, vect_goalobs, num_goalobs)"""
