import jax
import jax.numpy as jnp
import flax
from functools import partial


@flax.struct.dataclass
class Ball:
    """Defines balls in the environment"""
    pos: jnp.ndarray
    radius: jnp.ndarray

@flax.struct.dataclass
class Wall:
    """Defines walls in the environment"""
    start: jnp.ndarray
    end: jnp.ndarray
    unit_start_to_end: jnp.ndarray

@flax.struct.dataclass
class State:
    """Fully describes the system state"""
    ball: Ball
    obs: jnp.ndarray


def out(ball: Ball, down=0., up=1.):
    """ball in the square delimitted by down and up ?"""
    is_out = jnp.any(ball.pos < 0.)
    is_out = is_out | jnp.any(ball.pos > 1.)
    return is_out

def sample_ball_uniform(key, num_balls, x_limit, y_limit, r_min, r_max) -> Ball:
    """Samples balls randomly and uniformly"""
    scale = jnp.array([x_limit, y_limit])
    pos = jax.random.uniform(key, shape=(num_balls, 2))*scale
    radius = r_min + jax.random.uniform(key, shape=(num_balls,)) * (r_max - r_min)
    return Ball(pos=pos, radius=radius)

def sample_ball_uniform_pos(key, num_balls, x_limit, y_limit, radius) -> Ball:
    """Samples balls position randomly and uniformly"""
    min_values = jnp.stack((radius, radius), axis=-1)
    max_values = jnp.stack((x_limit-radius, y_limit-radius)).transpose()
    pos = jax.random.uniform(key, shape=(num_balls, 2))
    pos = min_values + pos*(max_values - min_values)
    return Ball(pos=pos, radius=radius)

def normalize(start: jnp.ndarray, end: jnp.ndarray) -> jnp.ndarray:
    """Normalizes the corresponding vector"""
    diff = end-start
    norm = jnp.linalg.norm(diff)
    return diff/norm    

def create_wall(start: jnp.ndarray, end: jnp.ndarray) -> Wall:
    """Instantiates a wall"""
    return Wall(start, end, normalize(start=start, end=end))

def sline_proj(pos: jnp.ndarray, w: Wall) -> jnp.ndarray:
    """Returns the projection of pos on the semi-line of origin w.start"""
    inner = jnp.dot(pos - w.start, w.unit_start_to_end)
    return w.start + jax.nn.relu(inner)*w.unit_start_to_end

def closest_point(pos: jnp.ndarray, w: Wall) -> jnp.ndarray:
    """Returns the closest point to pos on wall w."""
    slp = partial(sline_proj, w=w)
    inner_end = jnp.dot(pos - w.end, w.unit_start_to_end)
    return jax.lax.cond(inner_end > 0, lambda x: w.end, slp, pos)

def pen_res_bw(b: Ball, w: Wall) -> Ball:
    """Solves penetration problem : ball-wall"""
    closest = closest_point(b.pos, w)
    diff_unit = closest - b.pos
    dist_to_closest = jnp.linalg.norm(diff_unit)
    diff_unit = diff_unit / dist_to_closest
    return Ball(pos=b.pos - jax.nn.relu(b.radius - dist_to_closest)*diff_unit,
                radius=b.radius)

def pen_res_function_w(w: Wall):
    """Returns a function which solves the penetration problem 
    for one wall and multiple balls"""
    return jax.vmap(partial(pen_res_bw, w=w))

def pen_res_functions_w(w_list):
    """Returns a list of pen_res_function_w corresponding to each wall of the
    input list."""
    return [pen_res_function_w(w=w) for w in w_list]

def update_bw(balls: Ball, pen_res_fcts) -> Ball:
    """Return updated balls state"""
    new_balls = balls
    for fct in pen_res_fcts:
        new_balls = fct(new_balls)
    return new_balls

def pen_vect_bb(pos_1: jnp.ndarray, 
                radius_1: jnp.ndarray, 
                pos_2: jnp.ndarray, 
                radius_2: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
    """Returns the penetration vector between two balls"""
    diff = pos_1 - pos_2
    distance = jnp.linalg.norm(diff)
    pen_vect = jax.nn.relu((radius_1+radius_2)/distance - 1.)*diff
    return pen_vect

def pen_res_bb_pos(pos_1: jnp.ndarray, 
                   radius_1: jnp.ndarray, 
                   pos_2: jnp.ndarray, 
                   radius_2: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
    """Solves the penetration problem"""
    pen_vect = pen_vect_bb(pos_1, radius_1, pos_2, radius_2)
    pen_vect_2 = pen_vect / 2.
    return (pos_1 + pen_vect_2, pos_2 - pen_vect_2)

def get_ball_colliders(nb_balls: int):
    """Returns couples of possible collisions"""
    return [(i,j) for i in range(nb_balls) for j in range(i+1, nb_balls)]

def update_bb_indices(balls: Ball, index_1: int, index_2: int) -> Ball:
    """Updates positions of balls of indices index_1 and index_2"""
    new_pos_1, new_pos_2 = pen_res_bb_pos(balls.pos[index_1],
                                          balls.radius[index_1],
                                          balls.pos[index_2],
                                          balls.radius[index_2])
    new_balls_pos = balls.pos.at[index_1].set(new_pos_1)
    new_balls_pos = new_balls_pos.at[index_2].set(new_pos_2)
    return Ball(pos=new_balls_pos, radius=balls.radius)

def update_bb(balls: Ball, ball_colliders: list) -> Ball:
    """Updates balls positions through ball-to-ball collisions"""
    new_balls = balls
    for i,j in ball_colliders:
        new_balls = update_bb_indices(new_balls, i, j)
    return new_balls

def update_agent_pos(balls: Ball, 
                     dpos: jnp.ndarray, 
                     agent_index : int = 0) -> Ball:
    """Updates agent position."""
    new_agent_pos = balls.pos[agent_index] + dpos
    new_balls_pos = balls.pos.at[agent_index].set(new_agent_pos)
    return Ball(pos=new_balls_pos, radius=balls.radius)
