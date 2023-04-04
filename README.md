![licence](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Balzax
Game environments for reinforcement learning and open-ended learning composed of balls moving other balls and coded in JAX. 

## Installation

### Prerequisites
It is recommended to use a *conda environment* and follow the **GPU installation of [JAX](https://github.com/google/jax)**.

### Installation with conda
```
conda env create -f environment.yaml
conda activate balzax
pip install git+https://github.com/charlypg/Balzax
```

## Versions
- **0.1.0 :** *Balzax* environments
- **Over 0.1.1 :** *Balzax* environments with *terminated* and *truncated* instead of *done*, following the *OpenAI Gym* interface (versions over 0.25.2)
- **0.1.4 :** It is ***recommended*** to use version 0.1.4, as it is the most robust. It is also consistent with the *OpenAI Gym* interface (versions over 0.25.2).


## *Balzax* environments

### *Balzax* general framework

Originally, there are two JAX environments (state externally managed) : the *balls environments* ***BallsEnv*** and ***BallsEnvGoal*** (*goal-conditioned RL*), but ***Balzax*** also proposes ***Gym* wrappers** so as to apply most of *state-of-the-art* algorithms on it. In the implemented *Gym* wrappers, the environments vectorization is managed internally. *Balzax* also allows to implement custom environments thanks to the *abstract classes* ***BalzaxEnv*** and ***BalzaxGoalEnv***. 

### Balls environments 

## Examples

### Learning to push a ball to a desired position thanks to SAC+HER

The environment considered here is *BallsEnvGoal*. The goal of the agent is to move the small blue ball so as to guide the other blue ball to its goal position in red. After ***1 500 000 learning steps***, the following result is obtained using SAC+HER. 

https://user-images.githubusercontent.com/91844362/174491520-8bf9e437-6964-4cae-a90a-182c78f81844.mp4
