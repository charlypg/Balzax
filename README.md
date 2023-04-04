![licence](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Balzax

*Balzax* is an open source Python library for developing ***simple, vectorized and fast reinforcement learning environments*** thanks to JAX. It provides an API and already implemented game environments to test and compare reinforcement learning algorithms. 

## Installation

### Prerequisites
It is recommended to use a *conda environment* and follow the **GPU installation of [JAX](https://github.com/google/jax)** before *Balzax* installation.

### Installation with conda
```
conda env create -f environment.yaml
conda activate balzax
pip install git+https://github.com/charlypg/Balzax
```

## Versions
- **0.1.0 :** *Balzax* environments
- **Over 0.1.1 :** *Balzax* environments with *terminated* and *truncated* instead of *done*, following the *OpenAI Gym* interface (release 0.25.2).
- **0.1.4 :** It is ***recommended*** to use version 0.1.4, as it is the most robust. It is obviously consistent with the *OpenAI Gym* interface (versions over 0.25.2).


## *Balzax* environments

### *Balzax* general framework

*Balzax* provides a **general framework** to implement ***simple, vectorized and fast RL environments*** thanks to JAX automatic vectorization and *Just-In-Time (JIT)* compilation tools. 

Like in *OpenAI Gym*, your custom environment simply has to be an inherited class from ***BalzaxEnv*** (equivalent of *gym.Env*), or ***BalzaxGoalEnv*** (equivalent of *gym.GoalEnv*) for **goal-conditioned RL**. 

The key difference is that the state is externally managed as JAX is functional. It is not an attribute of the class environment as in *Gym* environments.

However, *Balzax* also provides a **Gym wrapper** for each type of environment (goal-oriented or not) so as to apply state-of-the-art algorithms and baselines.

*By the future*, *Balzax* environments might benefit from JAX automatic differentiation to implement **differentiable environments**.

### *Balls* environments 

Balls environments have given its name to *Balzax* (Balls+JAX). The idea is that the agent controls a small disk in a window, a ball, and its goal is to move some balls in the environment so as to reach a desired configuration. 

*See [**example 1**](#1-learning-to-push-a-ball-to-a-desired-position-thanks-to-sacher) and [**Balls environments description**](BALLS_ENVS.md) for more details.*

## Examples

### 1. Learning to push a ball to a desired position thanks to SAC+HER

The environment considered here is *BallsEnvGoal*. The goal of the agent is to move the small blue ball so as to guide the other blue ball to its goal position in red. The following trajectories are obtained using the policy learned thanks to SAC+HER. 

https://user-images.githubusercontent.com/91844362/174491520-8bf9e437-6964-4cae-a90a-182c78f81844.mp4

*See [**Balls environments description**](BALLS_ENVS.md) for more details.*
