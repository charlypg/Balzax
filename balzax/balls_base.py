import jax
import jax.numpy as jnp
from functools import partial

from balzax.structures import Ball
from balzax.structures import create_wall, get_ball_colliders, pen_res_functions_w
from balzax.structures import update_agent_pos, update_bb, update_bw
from balzax.structures import sample_ball_uniform_pos
from balzax.structures import out

from balzax.image_generation import balls_to_one_image


class BallsBase:
    def __init__(self, obs_type : str = 'position'): 
        self.num_balls = 4
        self.x_limit = 1.
        self.y_limit = 1.
        self.init_radius = (1. + jnp.arange(self.num_balls, dtype=jnp.float32))
        self.init_radius *= (min(self.x_limit, self.y_limit) / 20.)
        
        self.colors = 255*jnp.ones((self.num_balls,), dtype=jnp.int32)
        self.image_dim = 224
        
        self.ball_colliders = get_ball_colliders(self.num_balls)
        self.solve_bb_collisions = partial(update_bb, 
                                           ball_colliders=self.ball_colliders)
        
        self.walls = []
        self.walls.append(create_wall(start=jnp.array([0., 
                                                       0.]),
                                      end=jnp.array([self.x_limit, 
                                                     0.])))
        self.walls.append(create_wall(start=jnp.array([0., 
                                                       0.]),
                                      end=jnp.array([0., 
                                                     self.y_limit])))
        self.walls.append(create_wall(start=jnp.array([self.x_limit, 
                                                       0.]),
                                      end=jnp.array([self.x_limit, 
                                                     self.y_limit])))
        self.walls.append(create_wall(start=jnp.array([0., 
                                                       self.y_limit]),
                                      end=jnp.array([self.x_limit, 
                                                     self.y_limit])))
        
        self.solve_bw_collisions = partial(update_bw, 
                                           pen_res_fcts=pen_res_functions_w(self.walls))
        
        
        self.obs_type = obs_type
        self.obs_fcts = {
                        'position': self.get_pos, 
                        'image': self.get_image
                         }
        self.get_obs = self.obs_fcts.get(self.obs_type)
    
    def apply_update(self, 
                     balls: Ball, 
                     agent_pos_var: jnp.ndarray = jnp.zeros((2,))) -> Ball:
        """Update positions of balls"""
        new_balls = update_agent_pos(balls, agent_pos_var)
        new_balls = self.solve_bb_collisions(new_balls)
        new_balls = self.solve_bw_collisions(new_balls)
        return new_balls
    
    def get_dpos(self, action: jnp.ndarray) -> jnp.ndarray:
        """Returns a position variation from a velocity angle."""
        angle = jnp.pi * action
        return 0.02 * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    
    def reset_base(self, key):
        """Resets the game : new ball positions and new random key"""
        new_balls = sample_ball_uniform_pos(key, 
                                            self.num_balls, 
                                            self.x_limit, 
                                            self.y_limit, 
                                            self.init_radius)
        for _ in range(10):
            new_balls = self.solve_bb_collisions(new_balls)
            new_balls = self.solve_bw_collisions(new_balls)
        new_key = jax.random.split(key)[0]
        return new_balls, new_key
    
    def step_base(self, balls: Ball, action: jnp.ndarray) -> Ball:
        """Performs a game step"""
        dpos = self.get_dpos(action)
        new_balls = self.apply_update(balls, dpos)
        return new_balls
    
    def done_base(self, balls: Ball) -> bool:
        """Returns whether the game state is terminal or not"""
        return out(balls)
    
    def get_pos(self, balls: Ball) -> jnp.ndarray:
        """Returns positions from balls"""
        return balls.pos.flatten()
    
    def get_image(self, balls: Ball) -> jnp.ndarray:
        """Returns an image from balls"""
        return balls_to_one_image(balls,
                                  self.colors,
                                  self.image_dim)
