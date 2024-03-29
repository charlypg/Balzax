import jax
import jax.numpy as jnp
from time import time
from balzax.balls.balls_env_goal import BallsEnvGoal


@jax.jit
def vel(pulse, i):
    angle = jnp.sin(pulse * i)
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])


@jax.jit
def compute_done(terminated: jnp.ndarray, truncated: jnp.ndarray) -> jnp.ndarray:
    return jnp.logical_or(terminated, truncated)


print("TEST : BallsEnvGoal(obs_type='position')")
print()

env = BallsEnvGoal(obs_type="position", max_episode_steps=500)
jit_env_reset_done = jax.jit(env.reset_done)
jit_env_reset = jax.jit(env.reset)  # env.reset
jit_env_step = jax.jit(env.step)  # env.step

key = jax.random.PRNGKey(0)
NB_ITER_1 = 1
NB_ITER_2 = 10_000
assert NB_ITER_1 < NB_ITER_2
pulse = jnp.array(2 * jnp.pi / 200)
image_list = []

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

image_list.append(env.get_image(env_state.game_state))


t0 = time()
env_state = jit_env_step(env_state, jnp.zeros((2,)))
print("Time of first step (jit+exec) : {}s".format(time() - t0))
print("State of the environment : Timestep 1")
print(env_state)
print()

image_list.append(env.get_image(env_state.game_state))

t0 = time()
for i in range(NB_ITER_1):
    env_state = jit_env_step(env_state, vel(pulse, i))
    done = compute_done(env_state.terminated, env_state.truncated)
    env_state = jit_env_reset_done(env_state, done)
    # image_list.append(env.get_image(env_state.game_state))
print(
    "Rollout of {0} iterations (compiled step and reset_done) : {1}".format(
        NB_ITER_1, time() - t0
    )
)

t0 = time()
for i in range(NB_ITER_2):
    env_state = jit_env_step(env_state, vel(pulse, i))
    done = compute_done(env_state.terminated, env_state.truncated)
    env_state = jit_env_reset_done(env_state, done)
    # image_list.append(env.get_image(env_state.game_state))
print(
    "Rollout of {0} iterations (compiled step and reset_done) : {1}".format(
        NB_ITER_2, time() - t0
    )
)
print()
