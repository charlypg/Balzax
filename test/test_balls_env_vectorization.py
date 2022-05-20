import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from time import time

from balzax.balls_env import BallsEnv

OBS_TYPE = "image"
MAX_EPSISODE_STEPS = 3
SEED = 0
NUM_ENV = 3

NB_ITER = 21

ACTION_0 = jnp.zeros((NUM_ENV, 1))
ACTION_1 = jnp.ones((NUM_ENV, 1)) / 2.0

key = jax.random.PRNGKey(SEED)
keys = jax.random.split(key, num=NUM_ENV)

# print(keys)

env = BallsEnv(obs_type=OBS_TYPE, max_episode_steps=MAX_EPSISODE_STEPS)

vmap_env_reset_done = jax.jit(jax.vmap(env.reset_done))
vmap_env_reset = jax.jit(jax.vmap(env.reset))  # jax.vmap(env.reset)
vmap_env_step = jax.jit(jax.vmap(env.step))  # jax.vmap(env.step)

obs_list = []

print()
print("Observation type : {}".format(OBS_TYPE))
print("Seed : {}".format(SEED))
print("Number of envs : {}".format(NUM_ENV))
print()

t0 = time()
env_states = vmap_env_reset(keys)
print("Time of reset (jit+exec) : {}".format(time() - t0))
print()

t0 = time()
env_states = vmap_env_reset_done(env_states)
print("Time of reset_done (jit+exec) : {}".format(time() - t0))
print()

# print(env_states)

observations = env_states.obs
print("observations : {}".format(observations.shape))
print()

t0 = time()
env_states = vmap_env_reset(keys)
print("Time of reset (second time exec) : {}".format(time() - t0))
print()

obs_list.append(env_states.obs)

t0 = time()
env_states = vmap_env_step(env_states, ACTION_0)
print("First step (jit+exec) : {}".format(time() - t0))
print()

obs_list.append(env_states.obs)

t0 = time()
env_states = vmap_env_step(env_states, ACTION_1)
print("Second step (exec) : {}".format(time() - t0))
print()

obs_list.append(env_states.obs)

t0 = time()
for _ in range(NB_ITER):
    env_states = vmap_env_step(env_states, ACTION_1)
    env_states = vmap_env_reset_done(env_states)
    obs_list.append(env_states.obs)
print("{0} iterations in {1}s".format(NB_ITER, time() - t0))
print()

pulse = 2 * jnp.pi / NB_ITER * jnp.ones((NUM_ENV, 1))
t0 = time()
for i in range(NB_ITER):
    env_states = vmap_env_step(env_states, jnp.sin(pulse * i))
    env_states = vmap_env_reset_done(env_states)
    obs_list.append(env_states.obs)
print("{0} iterations in {1}s".format(NB_ITER, time() - t0))
print()

for i, obs in enumerate(obs_list):
    plt.figure(i)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Timestep {}".format(i))
    ax1.set_title("Env 0")
    ax1.imshow(obs[0], origin="lower")
    ax2.set_title("Env 1")
    ax2.imshow(obs[1], origin="lower")
    ax3.set_title("Env 2")
    ax3.imshow(obs[2], origin="lower")
