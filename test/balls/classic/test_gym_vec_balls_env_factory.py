import numpy as onp
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
from balzax.balls import gym_balls_env_factory

OBS_TYPE = "image"
NUM_BALLS = 4
NUM_ENVS = 3
SEED = 0
MAX_EPISODE_STEPS = 3
NB_ITER = 22
PULSE = 2 * onp.pi / NB_ITER * onp.ones((NUM_ENVS, 1))

gym_balls_env_factory(
    name="test_env",
    obs_type=OBS_TYPE,
    num_balls=NUM_BALLS,
    max_episode_steps=MAX_EPISODE_STEPS,
    num_envs=NUM_ENVS,
    seed=SEED,
    backend="gpu",
)

gym_env = gym.make("test_env")

t0 = time()
obs = gym_env.reset()
delta = time() - t0
print("gym_env.reset (first): {}".format(delta))
obs_list = [obs]

t0 = time()
for i in range(NB_ITER):
    angle = onp.sin(PULSE * i)
    cos = onp.cos(angle)
    sin = onp.sin(angle)
    action = onp.concatenate((cos, sin), axis=1)
    obs, reward, terminated, truncated, info = gym_env.step(action)
    done = onp.logical_or(terminated, truncated)
    obs = gym_env.reset_done(done=done, return_info=False)
    obs_list.append(obs)
delta = time() - t0
print("Rollout of {0} : {1}".format(NB_ITER, delta))

FRAMES = NB_ITER
fig, axs = plt.subplots(1, NUM_ENVS)


def animate_vect_goalobs(i):
    fig.suptitle("Timestep {}".format(i))
    for j in range(NUM_ENVS):
        axs[j].imshow(obs_list[i][j].squeeze(), origin="lower")


ani_goal = animation.FuncAnimation(fig, animate_vect_goalobs, frames=FRAMES)
FFwriter = animation.FFMpegWriter()
ani_goal.save(
    "test_gym_vec_balls_env.mp4",
    writer=FFwriter,
    progress_callback=lambda i, n: print(i),
)
