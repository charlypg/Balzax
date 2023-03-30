import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

from balzax.balls.balls_env import BallsEnv


@jax.jit
def vel(pulse, i):
    angle = jnp.sin(pulse * i)
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


print("TEST 1 : BallsEnv(obs_type='image')")
print()

env = BallsEnv(obs_type="image", num_balls=4, max_episode_steps=500)

jit_env_reset_done = jax.jit(env.reset_done)
jit_env_reset = jax.jit(env.reset)  # env.reset
jit_env_step = jax.jit(env.step)  # env.step

key = jax.random.PRNGKey(0)
NB_ITER = 50
pulse = jnp.array(2 * jnp.pi / 200)
image_list = []

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

image_list.append(env_state.obs)


t0 = time()
env_state = jit_env_step(env_state, jnp.zeros((2,)))
print("Time of first step (jit+exec) : {}s".format(time() - t0))
print("State of the environment : Timestep 1")
print(env_state)
print()

image_list.append(env_state.obs)

t0 = time()
for i in jnp.arange(NB_ITER):
    env_state = jit_env_step(env_state, vel(pulse, i))
    env_state = jit_env_reset_done(env_state)
    image_list.append(env_state.obs)

print(
    "Rollout of {0} iterations (compiled step and reset_done) : {1}".format(
        NB_ITER, time() - t0
    )
)
print()

FRAMES = NB_ITER
fig, axs = plt.subplots()


def animate_vect_goalobs(i):
    fig.suptitle("Timestep {}".format(i))
    axs.imshow(image_list[i].squeeze(), origin="lower")


ani_goal = animation.FuncAnimation(fig, animate_vect_goalobs, frames=FRAMES)
FFwriter = animation.FFMpegWriter()
ani_goal.save(
    "animation_rollout.mp4",
    writer=FFwriter,
    progress_callback=lambda i, n: print(i),
)
