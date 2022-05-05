import numpy as onp
import matplotlib.pyplot as plt
from time import time
 
from balzax.balls_env_goal import BallsEnvGoal
from balzax.wrapper import GoalGymVecWrapper


def plot_vect_goalobs(i : int, vect_goalobs : dict, num_goalobs : int):
    fig = plt.figure(i, constrained_layout=True)
    fig.suptitle('Timestep {}'.format(i))

    # create num_goalobs x 1 subfigs
    subfigs = fig.subfigures(nrows=num_goalobs, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle('Environment {}'.format(row))

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)
        for ax, field, images in zip(axs, 
                                    vect_goalobs.keys(), 
                                    vect_goalobs.values()):
            ax.imshow(images[row])
            ax.set_title(field)


OBS_TYPE = 'image'
NUM_ENVS = 3
SEED = 0
MAX_TIMESTEP = 5
NB_ITER = 21
PULSE = 2*onp.pi/NB_ITER * onp.ones((NUM_ENVS, 1))

env = BallsEnvGoal(obs_type=OBS_TYPE, max_timestep=MAX_TIMESTEP)
gym_env = GoalGymVecWrapper(env=env, num_envs=NUM_ENVS, seed=SEED)

t0 = time()
obs = gym_env.reset()
delta = time() - t0
print("gym_env.reset : {}".format(delta))
obs_list = [obs]

t0 = time()
for i in range(NB_ITER):
    action = onp.sin(PULSE * i)
    obs, reward, done, info = gym_env.step(action)
    obs = gym_env.reset_done()
    obs_list.append(obs)
delta = time() - t0
print("Rollout of {0} : {1}".format(NB_ITER, delta))

num_goalobs = min(2, NUM_ENVS)
for i, vect_goalobs in enumerate(obs_list):
    plot_vect_goalobs(i, vect_goalobs, num_goalobs)

"""
for i, obs in enumerate(obs_list):
    plt.figure(i)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Timestep {}'.format(i))
    ax1.set_title('Env 0')
    ax1.imshow(obs[0], origin='lower')
    ax2.set_title('Env 1')
    ax2.imshow(obs[1], origin='lower')
    ax3.set_title('Env 2')
    ax3.imshow(obs[2], origin='lower')"""
