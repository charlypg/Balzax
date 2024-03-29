import jax
import jax.numpy as jnp
import numpy as onp
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from balzax.balls.balls_env_goal import BallsEnvGoal
from balzax.wrapper import GoalGymVecWrapper


@jax.jit
def compute_done(terminated: jnp.ndarray, truncated: jnp.ndarray) -> jnp.ndarray:
    return jnp.logical_or(terminated, truncated)


OBS_TYPE = "position"
NUM_ENVS = 4
NUM_BALLS = 2
SEED = 0
MAX_EPISODE_STEPS = 30
NB_ITER = 90
PULSE = 2 * onp.pi / NB_ITER * onp.ones((NUM_ENVS, 1))

env = BallsEnvGoal(
    obs_type=OBS_TYPE, max_episode_steps=MAX_EPISODE_STEPS, num_balls=NUM_BALLS
)
gym_env = GoalGymVecWrapper(env=env, num_envs=NUM_ENVS, seed=SEED)

t0 = time()
obs = gym_env.reset(return_info=False)
delta = time() - t0
print("gym_env.reset : {}".format(delta))
obs_list = [obs]
info_list = []

rgb_image_list = [gym_env.render()]

t0 = time()
for i in tqdm(range(NB_ITER)):
    angle = onp.sin(PULSE * i)
    cos = onp.cos(angle)
    sin = onp.sin(angle)
    action = onp.concatenate((cos, sin), axis=1)
    obs, reward, terminated, truncated, info = gym_env.step(action)
    info_list.append(info)
    rgb_image_list.append(gym_env.render())
    done = compute_done(terminated, truncated)
    obs = gym_env.reset_done(done, return_info=False)
    obs_list.append(obs)
delta = time() - t0
print("Rollout of {0} : {1}".format(NB_ITER, delta))


FRAMES = NB_ITER
fig, axs = plt.subplots(NUM_ENVS, 1)


def animate_vect_goalobs(i):
    fig.suptitle("Timestep {}".format(i))
    for j in range(NUM_ENVS):
        axs[j].imshow(rgb_image_list[i][j].squeeze(), origin="lower")


ani_goal = animation.FuncAnimation(fig, animate_vect_goalobs, frames=FRAMES)
FFwriter = animation.FFMpegWriter()
ani_goal.save(
    "test_gym_vec_balls_env_goal_rendering.mp4",
    writer=FFwriter,
    progress_callback=lambda i, n: print(i),
)
