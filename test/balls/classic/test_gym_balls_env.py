import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

from balzax.balls.balls_env import BallsEnv
from balzax.wrapper import GymWrapper


env = BallsEnv(obs_type="image", max_episode_steps=500)
gym_env = GymWrapper(env=env, seed=0)

NB_ITER = 53
PULSE = onp.array(2 * onp.pi / NB_ITER)

t0 = time()
obs = gym_env.reset()
delta = time() - t0
print("gym_env.reset : {}".format(delta))
obs_list = [obs]
reward_list = []
info_list = []

t0 = time()
for i in range(NB_ITER):
    angle = onp.sin(PULSE * i)
    action = onp.array([onp.cos(-angle), onp.sin(-angle)])
    obs, reward, terminated, truncated, info = gym_env.step(action)
    reward_list.append(reward)
    info_list.append(info)
    done = onp.logical_or(terminated, truncated)
    obs = gym_env.reset_done(done=done, return_info=False)
    obs_list.append(obs)
delta = time() - t0
print("Rollout of {0} : {1}".format(NB_ITER, delta))

FRAMES = NB_ITER
fig, axs = plt.subplots()


def animate_vect_goalobs(i):
    fig.suptitle("Timestep {}".format(i))
    axs.imshow(obs_list[i].squeeze(), origin="lower")


ani_goal = animation.FuncAnimation(fig, animate_vect_goalobs, frames=FRAMES)
FFwriter = animation.FFMpegWriter()
ani_goal.save(
    "test_gym_balls_env.mp4",
    writer=FFwriter,
    progress_callback=lambda i, n: print(i),
)
