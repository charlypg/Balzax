import numpy as onp
import matplotlib.pyplot as plt
from time import time

from balzax.balls_env import BallsEnv
from balzax.wrapper import GymWrapper


env = BallsEnv(obs_type="image", max_episode_steps=500)
gym_env = GymWrapper(env=env, seed=0)

NB_ITER = 200
PULSE = onp.array(2 * onp.pi / NB_ITER)

t0 = time()
obs = gym_env.reset()
delta = time() - t0
print("gym_env.reset : {}".format(delta))
obs_list = [obs]
reward_list = []
done_list = []
info_list = []

t0 = time()
for i in range(NB_ITER):
    angle = onp.sin(PULSE * i)
    action = onp.array([onp.cos(-angle), onp.sin(-angle)])
    obs, reward, done, info = gym_env.step(action)
    reward_list.append(reward)
    done_list.append(done)
    info_list.append(info)
    obs = gym_env.reset_done()
    obs_list.append(obs)
delta = time() - t0
print("Rollout of {0} : {1}".format(NB_ITER, delta))

for i, image in enumerate(obs_list):
    plt.figure(i)
    plt.title("Timestep {}".format(i))
    plt.imshow(image, origin="lower")
