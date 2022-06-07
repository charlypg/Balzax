import jax
import jax.numpy as jnp
from functools import partial

from balzax.structures import Ball
from balzax.structures import create_wall, get_ball_colliders, pen_res_functions_w
from balzax.structures import update_agent_pos, update_bb, update_bw
from balzax.structures import out
from balzax.random_reset_base import compute_r, random_reset
from balzax.image_generation import balls_to_one_image


def clip_unit_circle(u: jnp.ndarray) -> jnp.ndarray:
    den = jnp.linalg.norm(u)
    den = 1 + jax.nn.relu(den - 1)
    return u / den


class BallsBase:
    NUM_BALLS_MAX = 4
    RAD_MIN = 0.035
    RAD_MAX = 0.065
    RAD_BASE = jnp.linspace(RAD_MIN, RAD_MAX, NUM_BALLS_MAX)
    VELOCITY_GAIN = 0.02

    def __init__(self, obs_type: str = "position", num_balls: int = 4):
        assert num_balls in range(1, 1 + BallsBase.NUM_BALLS_MAX)
        self.num_balls = num_balls
        self.length = 1.0
        self.x_limit = self.length
        self.y_limit = self.length
        self.epsilon = 1e-12
        self.nb_randsamp = 50
        self.r_max = compute_r(
            epsilon=self.epsilon, N=self.nb_randsamp, n=self.num_balls, L=self.length
        )
        assert BallsBase.RAD_MAX <= self.r_max
        self.init_radius = BallsBase.RAD_BASE[: self.num_balls]

        self.colors = 1.0 * jnp.ones((self.num_balls,), dtype=jnp.float32)
        self.image_dim = 224

        self.ball_colliders = get_ball_colliders(self.num_balls)
        self.solve_bb_collisions = partial(
            update_bb, ball_colliders=self.ball_colliders
        )

        self.walls = []
        self.walls.append(
            create_wall(start=jnp.array([0.0, 0.0]), end=jnp.array([self.x_limit, 0.0]))
        )
        self.walls.append(
            create_wall(start=jnp.array([0.0, 0.0]), end=jnp.array([0.0, self.y_limit]))
        )
        self.walls.append(
            create_wall(
                start=jnp.array([self.x_limit, 0.0]),
                end=jnp.array([self.x_limit, self.y_limit]),
            )
        )
        self.walls.append(
            create_wall(
                start=jnp.array([0.0, self.y_limit]),
                end=jnp.array([self.x_limit, self.y_limit]),
            )
        )

        self.solve_bw_collisions = partial(
            update_bw, pen_res_fcts=pen_res_functions_w(self.walls)
        )

        self.obs_type = obs_type
        self.obs_fcts = {"position": self.get_pos, "image": self.get_image}
        self.get_obs = self.obs_fcts.get(self.obs_type)

    def action_to_velocity(self, action: jnp.ndarray) -> jnp.ndarray:
        """Returns agent ball velocity from action vector"""
        return clip_unit_circle(action)

    def get_dpos(self, velocity: jnp.ndarray) -> jnp.ndarray:
        """Returns a position variation from velocity"""
        return BallsBase.VELOCITY_GAIN * velocity

    def step_base(self, balls: Ball, action: jnp.ndarray) -> Ball:
        """Performs a game step"""
        velocity = self.action_to_velocity(action)
        dpos = self.get_dpos(velocity)
        new_balls = update_agent_pos(balls, dpos)
        new_balls = self.solve_bb_collisions(new_balls)
        new_balls = self.solve_bw_collisions(new_balls)
        return new_balls

    def reset_base(self, key):
        """Resets the game : new ball positions and new random key"""
        new_balls, new_key = random_reset(
            key,
            num_balls=self.num_balls,
            x_limit=self.x_limit,
            y_limit=self.y_limit,
            radius=self.init_radius,
            N=self.nb_randsamp,
        )
        return new_balls, new_key

    def done_base(self, balls: Ball) -> jnp.ndarray:
        """Returns whether the game state is terminal or not"""
        return jnp.array([out(balls)], dtype=jnp.bool_)

    def get_pos(self, balls: Ball) -> jnp.ndarray:
        """Returns positions from balls"""
        return balls.pos.flatten()

    def get_image(self, balls: Ball) -> jnp.ndarray:
        """Returns an image from balls"""
        return balls_to_one_image(balls, self.colors, self.image_dim)

    def get_num_balls(self):
        """Getter : number of balls"""
        return self.num_balls

    def get_image_dim(self):
        """Getter : image dimensions"""
        return self.image_dim
