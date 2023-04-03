import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from balzax.balls.balls_env_goal import BallsEnvGoal


@jax.jit
def vel(pulse, i):
    angle = jnp.sin(pulse * i)
    return jnp.array([-jnp.sin(angle + jnp.pi / 2), -jnp.cos(angle + jnp.pi / 2)])


@jax.jit
def compute_done(terminated: jnp.ndarray, truncated: jnp.ndarray) -> jnp.ndarray:
    return jnp.logical_or(terminated, truncated)


print("TEST : BallsEnvGoal(obs_type='image')")
print()

env = BallsEnvGoal(obs_type="image", num_balls=3, max_episode_steps=50)

jit_env_reset_done = jax.jit(env.reset_done)
jit_env_reset = jax.jit(env.reset)  # env.reset
jit_env_step = jax.jit(env.step)  # env.step

key = jax.random.PRNGKey(0)
NB_ITER_1 = 1
NB_ITER_2 = 90
assert NB_ITER_1 < NB_ITER_2
pulse = 3 * jnp.array(2 * jnp.pi / (NB_ITER_2 - NB_ITER_1 + 1))
goalobs_list = []
metrics_list = []

t0 = time()
env_state = jit_env_reset(key)
print("Time to reset (jit+exec) : {}s".format(time() - t0))
print("State of the environment : Timestep 0")
print(env_state)
print()

t0 = time()
done = compute_done(env_state.terminated, env_state.truncated)
env_state = jit_env_reset_done(env_state, done)
print("Time to reset_done (jit+exec) : {}s".format(time() - t0))
print("State of the environment : Timestep 0")
print(env_state)
print()

goalobs_list.append(env_state.goalobs)
metrics_list.append(env_state.metrics)


t0 = time()
env_state = jit_env_step(env_state, jnp.zeros((2,)))
print("Time of first step (jit+exec) : {}s".format(time() - t0))
print("State of the environment : Timestep 1")
print(env_state)
print()

goalobs_list.append(env_state.goalobs)
metrics_list.append(env_state.metrics)

t0 = time()
for i in range(NB_ITER_1):
    env_state = jit_env_step(env_state, vel(pulse, i))
    metrics_list.append(env_state.metrics)
    done = compute_done(env_state.terminated, env_state.truncated)
    env_state = jit_env_reset_done(env_state, done)
    goalobs_list.append(env_state.goalobs)
print(
    "Rollout of {0} iterations (compiled step and reset_done) : {1}".format(
        NB_ITER_1, time() - t0
    )
)

t0 = time()
for i in range(1, NB_ITER_2):
    env_state = jit_env_step(env_state, vel(pulse, i))
    metrics_list.append(env_state.metrics)
    done = compute_done(env_state.terminated, env_state.truncated)
    env_state = jit_env_reset_done(env_state, done)
    goalobs_list.append(env_state.goalobs)
print(
    "Rollout of {0} iterations (compiled step and reset_done) : {1}".format(
        NB_ITER_2, time() - t0
    )
)
print()


FRAMES = NB_ITER_1 + NB_ITER_2
fig, axs = plt.subplots(1, 3)


def animate_vect_goalobs(i):
    fig.suptitle("Timestep {}".format(i))
    for k, (field, value) in enumerate(
        zip(goalobs_list[i].keys(), goalobs_list[i].values())
    ):
        axs[k].imshow(value.squeeze(), origin="lower")
        axs[k].set_title(field)


ani_goal = animation.FuncAnimation(fig, animate_vect_goalobs, frames=FRAMES)
FFwriter = animation.FFMpegWriter()
ani_goal.save(
    "test_balls_env_goal_image.mp4",
    writer=FFwriter,
    progress_callback=lambda i, n: print(i),
)
