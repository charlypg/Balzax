import numpy as onp
import jax
import jax.numpy as jnp
from functools import partial

from balzax.structures import Ball
from balzax.structures import sample_ball_uniform_pos


def f(p_M, n):
    return onp.sqrt(p_M / (2 * n * (n - 1)))


def lininv(x):
    return x / (1 + x)


def compute_l(p_M, n, L):
    return lininv(f(p_M, n)) * L


def compute_N(p_M, epsilon):
    return 1 + int(onp.log(epsilon) / onp.log(p_M))


def compute_p_M(epsilon, N):
    return epsilon ** (1 / N)


def compute_r(epsilon, N, n, L):
    assert n >= 1
    max_r = L / 10.0
    if n == 1:
        return max_r
    p_M = compute_p_M(epsilon, N)
    r = compute_l(p_M, n, L) / 2
    return min(r, max_r)


def sample_balls(keys, num_balls, x_limit, y_limit, radius):
    return jax.vmap(
        partial(
            sample_ball_uniform_pos,
            num_balls=num_balls,
            x_limit=x_limit,
            y_limit=y_limit,
            radius=radius,
        )
    )(keys)


def elem_dist_cost(ball_1: Ball, ball_2: Ball):
    r_part = (ball_1.radius + ball_2.radius) ** 2
    d_part = jnp.sum((ball_1.pos - ball_2.pos) ** 2)
    cost = jax.nn.relu(r_part - d_part) ** 2
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
    return pos + radius - 1.0


def up_left_cost(balls: Ball):
    terms = jax.nn.relu(-subs_pos_rad(balls.pos, balls.radius)) ** 2
    return jnp.sum(terms)


def low_right_cost(balls: Ball):
    terms = jax.nn.relu(add_m1_pos_rad(balls.pos, balls.radius)) ** 2
    return jnp.sum(terms)


def cost(balls: Ball):
    costs = jnp.array(
        [
            collision_cost(balls),
            constant_cost(balls),
            up_left_cost(balls),
            low_right_cost(balls),
        ]
    )
    return jnp.sum(costs)


def costs(balls):
    return jax.vmap(cost)(balls)


def random_reset(key, num_balls, x_limit, y_limit, radius, N):
    new_key = jax.random.split(key)[0]
    keys = jax.random.split(key, num=N)
    balls = sample_balls(keys, num_balls, x_limit, y_limit, radius)
    cc = costs(balls)
    i_min = jnp.argmin(cc)
    ball_samp = Ball(pos=balls.pos[i_min], radius=balls.radius[i_min])
    return ball_samp, new_key


def test_random_reset(nb_iter, num_balls, x_limit, y_limit, radius, N):
    @jax.jit
    def aux(key):
        ball_samp, new_key = random_reset(key, num_balls, x_limit, y_limit, radius, N)
        return ball_samp, new_key

    key = jax.random.PRNGKey(0)
    cost_list = []
    for _ in range(nb_iter):
        ball_samp, key = aux(key)
        cost_list.append(cost(ball_samp))
    return cost_list
