import numpy as onp
import matplotlib.pyplot as plt
from time import time

from balzax.balls_env import BallsEnv
from balzax.wrapper import GymVecWrapper

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
    obs = gym_env.reset_done()
    obs_list.append(obs)
delta = time() - t0
print("Rollout of {0} : {1}".format(NB_ITER, delta))

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
