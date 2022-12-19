import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from balzax.balls_env_goal import BallsEnvGoal
from balzax.wrapper import GoalGymVecWrapper
from balzax.structures import Ball
from balzax.image_generation import balls_to_one_image


def goal_to_ball(gym_env: GoalGymVecWrapper, goal):
    balls = gym_env.env_state.game_state
    goal_radius = balls.radius
    goal_pos = goal.reshape(balls.pos.shape)
    pos = jnp.concatenate([balls.pos, goal_pos])
    radius = jnp.concatenate([balls.radius, goal_radius])
    return Ball(pos=pos, radius=radius)


def custom_render(gym_env: GoalGymVecWrapper, goal, color):
    ball = goal_to_ball(gym_env, goal)
    rgb_image = balls_to_one_image(ball, color).squeeze()
    return rgb_image


OBS_TYPE = "position"
NUM_ENVS = 1
NUM_BALLS = 1
SEED = 0
MAX_EPISODE_STEPS = 50
NB_ITER = 220
PULSE = 2 * onp.pi / NB_ITER * onp.ones((NUM_ENVS, 1))

COLOR = onp.array([[0, 0, 255], [255, 0, 0]], dtype=onp.int32)

env = BallsEnvGoal(
    obs_type=OBS_TYPE, max_episode_steps=MAX_EPISODE_STEPS, num_balls=NUM_BALLS
)
gym_env = GoalGymVecWrapper(env=env, num_envs=NUM_ENVS, seed=SEED)

t0 = time()
obs = gym_env.reset()
delta = time() - t0
print("gym_env.reset : {}".format(delta))
obs_list = [obs]
info_list = []

rgb_image_list = [custom_render(gym_env, obs.get("desired_goal"), COLOR)]

t0 = time()
for i in tqdm(range(NB_ITER)):
    angle = onp.sin(PULSE * i)
    cos = onp.cos(angle)
    sin = onp.sin(angle)
    action = onp.concatenate((cos, sin), axis=1)
    obs, reward, terminated, truncated, info = gym_env.step(action)
    info_list.append(info)
    rgb_image_list.append(custom_render(gym_env, obs.get("desired_goal"), COLOR))
    obs = gym_env.reset_done()
    obs_list.append(obs)
delta = time() - t0
print("Rollout of {0} : {1}".format(NB_ITER, delta))

for i, rgb_image in enumerate(rgb_image_list):
    plt.figure(i)
    plt.title("Timestep {}".format(i))
    plt.imshow(rgb_image, origin="lower")
