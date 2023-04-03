import jax
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
import gym
from balzax.balls.gym_balls_env_goal import gym_balls_env_goal_factory


@jax.jit
def compute_done(terminated: jnp.ndarray, truncated: jnp.ndarray) -> jnp.ndarray:
    return jnp.logical_or(terminated, truncated)


OBS_TYPE = "image"
NUM_BALLS = 3
NUM_ENVS = 3
SEED = 0
MAX_EPISODE_STEPS = 5
NB_ITER = 21
PULSE = 2 * onp.pi / NB_ITER * onp.ones((NUM_ENVS, 1))

gym_balls_env_goal_factory(
    name="test_goal_env",
    obs_type=OBS_TYPE,
    num_balls=NUM_BALLS,
    goal_projection="identity",
    max_episode_steps=MAX_EPISODE_STEPS,
    num_envs=NUM_ENVS,
    seed=SEED,
    backend="gpu",
)

gym_env = gym.make("test_goal_env")

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


FRAMES = NB_ITER
fig, axs = plt.subplots(NUM_ENVS, 3)


def animate_vect_goalobs(i):
    fig.suptitle("Timestep {}".format(i))
    for j in range(NUM_ENVS):
        for k, (field, value) in enumerate(
            zip(obs_list[i].keys(), obs_list[i].values())
        ):
            axs[j, k].imshow(value[j].squeeze(), origin="lower")
            axs[j, k].set_title(field)


ani_goal = animation.FuncAnimation(fig, animate_vect_goalobs, frames=FRAMES)
FFwriter = animation.FFMpegWriter()
ani_goal.save(
    "test_gym_vec_balls_env_goal_factory.mp4",
    writer=FFwriter,
    progress_callback=lambda i, n: print(i),
)
