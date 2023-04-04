#!/bin/bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1

python test_balls_env_image.py
python test_balls_env_position.py
python test_balls_env_vectorization.py
python test_gym_vec_balls_env.py
python test_gym_vec_balls_env_factory.py