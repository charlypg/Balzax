import jax
import jax.numpy as jnp
from functools import partial

from structures import Ball

def scale_ball(ball: Ball, scale) -> Ball:
    """Performs homothety"""
    return Ball(pos=scale*ball.pos, radius=scale*ball.radius)

def isin_ball(y: jnp.ndarray, x: jnp.ndarray, ball: Ball):
    """(x,y) in ball ?"""
    sq = jnp.array([x, y])
    sq = sq - ball.pos
    sq = sq*sq
    return (jnp.sum(sq) <= ball.radius*ball.radius)

def isin_ball_vect_y(x: jnp.ndarray, y_vect: jnp.ndarray, ball: Ball):
    """Batch version over y axis of isin_ball"""
    return jax.vmap(partial(isin_ball, x=x, ball=ball))(y_vect)

def isin_ball_mat(x_vect: jnp.ndarray, y_vect: jnp.ndarray, ball: Ball):
    """Batch version over x axis of isin_ball_vect_y in order to obtain a 
    boolean image."""
    return jax.vmap(partial(isin_ball_vect_y, y_vect=y_vect, ball=ball))(x_vect)

def ball_to_imbool(reg_ball: Ball,  
                  image_dim: int = 224):
    """Boolean image of a ball"""
    x_vect = jnp.arange(image_dim)
    y_vect = jnp.arange(image_dim)
    image_ball = scale_ball(reg_ball, image_dim)
    return isin_ball_mat(x_vect, y_vect, image_ball)

def ball_to_image(reg_ball: Ball,
                  color: jnp.ndarray = jnp.array([255, 0, 0], dtype=jnp.int32),
                  image_dim: int = 224):
    """Returns an image of a colored ball."""
    mask = jnp.expand_dims(ball_to_imbool(reg_ball, image_dim), 
                              axis=-1)
    return mask * color

def balls_to_images(reg_ball: Ball,
                    color: jnp.ndarray,
                    image_dim: int = 224):
    """Returns a batch of colored balls"""
    return jax.vmap(ball_to_image, in_axes=[0, 0, None])(reg_ball, color, image_dim)

def balls_to_one_image(reg_ball: Ball,
                       color: jnp.ndarray,
                       image_dim: int = 224):
    """Returns one image with all batched balls"""
    return jnp.clip(jnp.sum(balls_to_images(reg_ball, color, image_dim), axis=0),
                    0,
                    255)
    