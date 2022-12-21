import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time

from balzax.balls_env_goal import BallsEnvGoal


def vel(pulse, i):
    angle = jnp.sin(pulse * i)
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])

vmap_vel = jax.jit(jax.vmap(vel))

@jax.jit
def compute_done(terminated: jnp.ndarray, truncated: jnp.ndarray) -> jnp.ndarray:
    return jnp.logical_or(terminated, truncated)

def plot_vect_goalobs(i: int, vect_goalobs: dict, num_goalobs: int):
    fig = plt.figure(i, constrained_layout=True)
    fig.suptitle("Timestep {}".format(i))

    # create num_goalobs x 1 subfigs
    subfigs = fig.subfigures(nrows=num_goalobs, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle("Environment {}".format(row))

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)
        for ax, field, images in zip(axs, vect_goalobs.keys(), vect_goalobs.values()):
            ax.imshow(images[row])
            ax.set_title(field)


OBS_TYPE = "image"
SEED = 0
NUM_ENV = 3
MAX_EPISODE_STEPS = 5

NB_ITER_1 = 1
NB_ITER_2 = 21
assert NB_ITER_1 < NB_ITER_2

ACTION_0 = jnp.zeros((NUM_ENV, 2))
ACTION_1 = jnp.ones((NUM_ENV, 2))

key = jax.random.PRNGKey(SEED)
keys = jax.random.split(key, num=NUM_ENV)

env = BallsEnvGoal(obs_type=OBS_TYPE, max_episode_steps=MAX_EPISODE_STEPS)

vmap_env_reset = jax.jit(jax.vmap(env.reset))
vmap_env_reset_done = jax.jit(jax.vmap(env.reset_done))
vmap_env_step = jax.jit(jax.vmap(env.step))

goalobs_list = []
metrics_list = []

print()
print("Observation type : {}".format(OBS_TYPE))
print("Seed : {}".format(SEED))
print("Number of envs : {}".format(NUM_ENV))
print()

t0 = time()
env_states = vmap_env_reset(keys)
print("Time of reset (jit+exec) : {}".format(time() - t0))
print()

for key_goal, value_goal in zip(env_states.goalobs.keys(), env_states.goalobs.values()):
    print("{0} shape : {1}".format(key_goal, value_goal.shape))
print()

goalobs_list.append(env_states.goalobs)
metrics_list.append(env_states.metrics)

t0 = time()
env_states = vmap_env_step(env_states, ACTION_0)
print("First step (jit+exec) : {}".format(time() - t0))
print()

goalobs_list.append(env_states.goalobs)
metrics_list.append(env_states.metrics)

t0 = time()
env_states = vmap_env_step(env_states, ACTION_1)
print("Second step (exec) : {}".format(time() - t0))
print()

goalobs_list.append(env_states.goalobs)
metrics_list.append(env_states.metrics)

t0 = time()
for _ in range(NB_ITER_1):
    env_states = vmap_env_step(env_states, ACTION_1)
    metrics_list.append(env_states.metrics)
    done = compute_done(env_states.terminated, env_states.truncated)
    env_states = vmap_env_reset_done(env_states, done)
    goalobs_list.append(env_states.goalobs)
print("{0} iterations in {1}s".format(NB_ITER_1, time() - t0))
print()

pulse = 2 * jnp.pi / NB_ITER_2 * jnp.ones((NUM_ENV,))
t0 = time()
for i in range(NB_ITER_1, NB_ITER_1 + NB_ITER_2):
    env_states = vmap_env_step(env_states, vmap_vel(pulse, i * jnp.ones((NUM_ENV,))))
    metrics_list.append(env_states.metrics)
    done = compute_done(env_states.terminated, env_states.truncated)
    env_states = vmap_env_reset_done(env_states, done)
    goalobs_list.append(env_states.goalobs)
print("{0} iterations in {1}s".format(NB_ITER_2, time() - t0))
print()

num_goalobs = min(2, NB_ITER_2)
for i, vect_goalobs in enumerate(goalobs_list):
    plot_vect_goalobs(i, vect_goalobs, num_goalobs)
