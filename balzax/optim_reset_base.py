import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from functools import partial
from balzax.structures import Ball

def elem_dist_cost(ball_1: Ball, ball_2: Ball):
    r_part = (ball_1.radius + ball_2.radius)**2
    d_part = jnp.sum((ball_1.pos - ball_2.pos)**2)
    cost = jax.nn.relu(r_part - d_part)**2
    return cost

def line_dist_cost(ball_2: Ball, balls_1: Ball):
    return jax.vmap(partial(elem_dist_cost, ball_2=ball_2))(balls_1)

def squa_dist_cost(balls: Ball):
    return jax.vmap(partial(line_dist_cost, balls_1=balls))(balls)

def collision_cost(balls: Ball):
    return jnp.sum(squa_dist_cost(balls))

def constant_cost(balls: Ball):
    return -16 * jnp.sum(balls.radius**4)

@jax.vmap
def subs_pos_rad(pos: jnp.ndarray, radius: jnp.ndarray):
    return pos - radius

@jax.vmap
def add_m1_pos_rad(pos: jnp.ndarray, radius: jnp.ndarray):
    return pos + radius - 1.

def up_left_cost(balls: Ball):
    terms = jax.nn.relu( - subs_pos_rad(balls.pos, balls.radius) )**2
    return jnp.sum(terms)

def low_right_cost(balls: Ball):
    terms = jax.nn.relu(add_m1_pos_rad(balls.pos, balls.radius))**2
    return jnp.sum(terms)
    
def cost(balls: Ball):
    costs = jnp.array([collision_cost(balls),
                       constant_cost(balls),
                       up_left_cost(balls),
                       low_right_cost(balls)])
    return jnp.sum(costs)

def sample_uniform_pos(key, num_balls, x_limit, y_limit, radius):
    """Samples balls position randomly and uniformly"""
    min_values = jnp.stack((radius, radius), axis=-1)
    max_values = jnp.stack((x_limit-radius, y_limit-radius)).transpose()
    pos = jax.random.uniform(key, shape=(num_balls, 2))
    pos = min_values + pos*(max_values - min_values)
    return pos

def pos_cost(position: jnp.ndarray, radius: jnp.ndarray, num_balls: int):
    pos = position.reshape(num_balls, 2)
    ball = Ball(pos, radius)
    return cost(ball)

def solve_reset(pos_0: jnp.ndarray, radius: jnp.ndarray):
    opt = minimize(pos_cost, 
                   x0=pos_0, 
                   args=(radius, radius.shape[0]),
                   method='BFGS')
    return opt.x, opt.fun
