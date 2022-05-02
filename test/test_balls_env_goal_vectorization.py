#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time

from balzax.env import GoalObs
from balzax.balls_env_goal import BallsEnvGoal

def plot_vect_goalobs(i : int, vect_goalobs : GoalObs, num_goalobs : int):
    plt.figure(i)
    fig, axs = plt.subplots(num_goalobs, 3)
    fig.suptitle('Timestep {}'.format(i))
    for k, (key_goal, value_goal) in enumerate(zip(vect_goalobs.keys(), 
                                                   vect_goalobs.values())):
        for j in range(num_goalobs):
            axs[j, k].imshow(value_goal[j], origin='lower')
            axs[j, k].set_title('Env {0} : {1}'.format(j, key_goal))


OBS_TYPE = 'image'
SEED = 0
NUM_ENV = 100
MAX_TIMESTEP = 5

NB_ITER_1 = 1
NB_ITER_2 = 1000
assert NB_ITER_1 < NB_ITER_2

ACTION_0 = jnp.zeros((NUM_ENV,))
ACTION_1 = jnp.ones((NUM_ENV,))/2.

key = jax.random.PRNGKey(SEED)
keys = jax.random.split(key, num=NUM_ENV)

env = BallsEnvGoal(obs_type=OBS_TYPE, max_timestep=MAX_TIMESTEP)

vmap_env_reset = jax.jit(jax.vmap(env.reset))
vmap_env_reset_done = jax.jit(jax.vmap(env.reset_done))
vmap_env_step = jax.jit(jax.vmap(env.step))  

goalobs_list = []

print()
print("Observation type : {}".format(OBS_TYPE))
print("Seed : {}".format(SEED))
print("Number of envs : {}".format(NUM_ENV))
print()

t0 = time()
env_states = vmap_env_reset(keys)
print("Time of reset (jit+exec) : {}".format(time()-t0))
print()

for key_goal, value_goal in zip(env_states.goalobs.keys(), 
                                env_states.goalobs.values()):
    print("{0} shape : {1}".format(key_goal, value_goal.shape))
print()

goalobs_list.append(env_states.goalobs)

t0 = time()
env_states = vmap_env_step(env_states, ACTION_0)
print("First step (jit+exec) : {}".format(time()-t0))
print()

goalobs_list.append(env_states.goalobs)

t0 = time()
env_states = vmap_env_step(env_states, ACTION_1)
print("Second step (exec) : {}".format(time()-t0))
print()

goalobs_list.append(env_states.goalobs)

t0 = time()
for _ in range(NB_ITER_1):
    env_states = vmap_env_step(env_states, ACTION_1)
    env_states = vmap_env_reset_done(env_states)
    #goalobs_list.append(env_states.goalobs)
print("{0} iterations in {1}s".format(NB_ITER_1, time()-t0))
print()

pulse = 2*jnp.pi / NB_ITER_2 * jnp.ones((NUM_ENV,))
t0 = time()
for i in range(NB_ITER_1, NB_ITER_2):
    env_states = vmap_env_step(env_states, jnp.sin(pulse*i))
    env_states = vmap_env_reset_done(env_states)
    #goalobs_list.append(env_states.goalobs)
print("{0} iterations in {1}s".format(NB_ITER_2, time()-t0))
print()
"""
num_goalobs = min(3, NB_ITER_2)
for i, vect_goalobs in enumerate(goalobs_list):
    plot_vect_goalobs(i, vect_goalobs, num_goalobs)"""