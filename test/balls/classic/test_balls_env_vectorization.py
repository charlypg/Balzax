import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from time import time

from balzax.balls.balls_env import BallsEnv


def vel(pulse, i):
    angle = jnp.sin(pulse * i)
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


vmap_vel = jax.jit(jax.vmap(vel))

OBS_TYPE = "image"
MAX_EPSISODE_STEPS = 3
SEED = 0
NUM_ENV = 3

NB_ITER = 57

ACTION_0 = jnp.zeros((NUM_ENV, 2))
ACTION_1 = jnp.ones((NUM_ENV, 2))

key = jax.random.PRNGKey(SEED)
keys = jax.random.split(key, num=NUM_ENV)

# print(keys)

env = BallsEnv(obs_type=OBS_TYPE, max_episode_steps=MAX_EPSISODE_STEPS)

vmap_env_reset_done = jax.jit(jax.vmap(env.reset_done))
vmap_env_reset = jax.jit(jax.vmap(env.reset))
vmap_env_step = jax.jit(jax.vmap(env.step))

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

pulse = 2 * jnp.pi / NB_ITER * jnp.ones((NUM_ENV,))
t0 = time()
for i in range(NB_ITER):
    env_states = vmap_env_step(env_states, vmap_vel(pulse, i * jnp.ones((NUM_ENV,))))
    env_states = vmap_env_reset_done(env_states)
    obs_list.append(env_states.obs)
print("{0} iterations in {1}s".format(NB_ITER, time() - t0))
print()

FRAMES = NB_ITER
fig, axs = plt.subplots(1, NUM_ENV)


def animate_vect_goalobs(i):
    fig.suptitle("Timestep {}".format(i))
    for j in range(NUM_ENV):
        axs[j].imshow(obs_list[i][j].squeeze(), origin="lower")


ani_goal = animation.FuncAnimation(fig, animate_vect_goalobs, frames=FRAMES)
FFwriter = animation.FFMpegWriter()
ani_goal.save(
    "animation_rollout.mp4",
    writer=FFwriter,
    progress_callback=lambda i, n: print(i),
)
