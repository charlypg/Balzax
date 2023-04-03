#!/bin/bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2

echo "TEST: test_balls_env_goal_image.py"
python test_balls_env_goal_image.py
echo
echo "TEST: test_balls_env_goal_position.py"
python test_balls_env_goal_position.py
echo
echo "TEST: test_balls_env_goal_vectorization.py"
python test_balls_env_goal_vectorization.py
echo
echo "TEST: test_gym_vec_balls_env_goal_rendering.py"
python test_gym_vec_balls_env_goal_rendering.py
echo
echo "TEST: test_gym_vec_balls_env_goal.py"
python test_gym_vec_balls_env_goal.py
echo
echo "TEST: test_gym_vec_balls_env_goal_factory.py"
python test_gym_vec_balls_env_goal_factory.py
