import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time

from balzax.balls_env_goal import BallsEnvGoal


def plot_goalobs(i, goalobs: dict):
    plt.figure(i)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Timestep {}".format(i))
    ax1.set_title("achieved goal")
    ax1.imshow(goalobs.get("achieved_goal"), origin="lower")
    ax2.set_title("desired goal")
    ax2.imshow(goalobs.get("desired_goal"), origin="lower")
    ax3.set_title("observation")
    ax3.imshow(goalobs.get("observation"), origin="lower")


print("TEST : BallsEnvGoal(obs_type='image')")
print()

env = BallsEnvGoal(obs_type="image", max_timestep=50)

jit_env_reset_done = jax.jit(env.reset_done)
jit_env_reset = jax.jit(env.reset)  # env.reset
jit_env_step = jax.jit(env.step)  # env.step

key = jax.random.PRNGKey(0)
nb_iter_1 = 1
nb_iter_2 = 200
assert nb_iter_1 < nb_iter_2
pulse = jnp.array([2 * jnp.pi / 200])
goalobs_list = []
metrics_list = []

t0 = time()
env_state = jit_env_reset(key)
print("Time to reset (jit+exec) : {}s".format(time() - t0))
print("State of the environment : Timestep 0")
print(env_state)
print()

t0 = time()
env_state = jit_env_reset_done(env_state)
print("Time to reset_done (jit+exec) : {}s".format(time() - t0))
print("State of the environment : Timestep 0")
print(env_state)
print()

goalobs_list.append(env_state.goalobs)
metrics_list.append(env_state.metrics)


t0 = time()
env_state = jit_env_step(env_state, jnp.array([0.0]))
print("Time of first step (jit+exec) : {}s".format(time() - t0))
print("State of the environment : Timestep 1")
print(env_state)
print()

goalobs_list.append(env_state.goalobs)
metrics_list.append(env_state.metrics)

t0 = time()
for i in range(nb_iter_1):
    env_state = jit_env_step(env_state, jnp.sin(pulse * i))
    metrics_list.append(env_state.metrics)
    env_state = jit_env_reset_done(env_state)
    goalobs_list.append(env_state.goalobs)
print(
    "Rollout of {0} iterations (compiled step and reset_done) : {1}".format(
        nb_iter_1, time() - t0
    )
)

t0 = time()
for i in range(1, nb_iter_2):
    env_state = jit_env_step(env_state, jnp.sin(pulse * i))
    metrics_list.append(env_state.metrics)
    env_state = jit_env_reset_done(env_state)
    goalobs_list.append(env_state.goalobs)
print(
    "Rollout of {0} iterations (compiled step and reset_done) : {1}".format(
        nb_iter_2, time() - t0
    )
)
print()

for i, goalobs in enumerate(goalobs_list):
    plot_goalobs(i, goalobs)
