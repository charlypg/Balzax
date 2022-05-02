import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from time import time
 
from balzax.balls_env_goal import BallsEnvGoal

print("TEST : BallsEnvGoal(obs_type='position')")
print()
 
env = BallsEnvGoal(obs_type='position', max_timestep=500)
jit_env_reset_done = jax.jit(env.reset_done)
jit_env_reset = jax.jit(env.reset) # env.reset
jit_env_step = jax.jit(env.step) # env.step
 
key = jax.random.PRNGKey(0)
nb_iter_1 = 1
nb_iter_2 = 10_000
assert nb_iter_1 < nb_iter_2
pulse = 2*jnp.pi/200
image_list = []
 
t0 = time()
env_state = jit_env_reset(key)
print("Time to reset (jit+exec) : {}s".format(time()-t0))
print("State of the environment : Timestep 0")
print(env_state)
print()

t0 = time()
env_state = jit_env_reset_done(env_state)
print("Time to reset_done (jit+exec) : {}s".format(time()-t0))
print("State of the environment : Timestep 0")
print(env_state)
print()
 
image_list.append(env.get_image(env_state.game_state))
 
 
t0 = time()
env_state = jit_env_step(env_state, jnp.array(0.))
print("Time of first step (jit+exec) : {}s".format(time()-t0))
print("State of the environment : Timestep 1")
print(env_state)
print()
 
image_list.append(env.get_image(env_state.game_state))

t0 = time()
for i in range(nb_iter_1):
    env_state = jit_env_step(env_state, jnp.sin(pulse*i))
    env_state = jit_env_reset_done(env_state)
    #image_list.append(env.get_image(env_state.game_state))
print("Rollout of {0} iterations (compiled step and reset_done) : {1}".format(nb_iter_1, 
                                                                          time()-t0))
 
t0 = time()
for i in range(nb_iter_2):
    env_state = jit_env_step(env_state, jnp.sin(pulse*i))
    env_state = jit_env_reset_done(env_state)
    #image_list.append(env.get_image(env_state.game_state))
print("Rollout of {0} iterations (compiled step and reset_done) : {1}".format(nb_iter_2, 
                                                                          time()-t0))
print()

for i, image in enumerate(image_list):
    plt.figure(i)
    plt.title('Timestep {}'.format(i))
    plt.imshow(image, origin='lower')
