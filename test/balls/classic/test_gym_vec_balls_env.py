import numpy as onp
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from balzax.balls.balls_env import BallsEnv
from balzax.wrapper import GymVecWrapper

# TODO: Test actually used gym environments

OBS_TYPE = "image"
NUM_ENVS = 3
SEED = 0
MAX_EPISODE_STEPS = 3
NB_ITER = 22
PULSE = 2 * onp.pi / NB_ITER * onp.ones((NUM_ENVS, 1))

env = BallsEnv(obs_type=OBS_TYPE, max_episode_steps=MAX_EPISODE_STEPS)
gym_env = GymVecWrapper(env=env, num_envs=NUM_ENVS, seed=SEED)

t0 = time()
obs = gym_env.reset()
delta = time() - t0
print("gym_env.reset : {}".format(delta))
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
