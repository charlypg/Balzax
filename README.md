![licence](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Balzax
Game environments for reinforcement learning and open-ended learning composed of balls moving other balls and coded in JAX. 

## Installation

### CPU
With conda :
```
conda env create -f environment_cpu.yaml
conda activate balzax_cpu
pip install git+https://github.com/charlypg/Balzax
```

### GPU
As before, it is recommended to use a *conda environment*.

*Follow the instructions for the **GPU installation of [JAX](https://github.com/google/jax)**.*

Then install *Balzax dependencies* : 
- flax>=0.4.1
- gym>=0.22.0
- numpy
- matplotlib

Finally, install *Balzax* :
```
pip install git+https://github.com/charlypg/Balzax
```

## Versions
- **0.1.0 :** Balzax environments
- **Over 0.1.1 :** Balzax environments with *terminated* and *truncated* instead of *done*, following the new OpenAI Gym interface (version 0.25.2)


## *Balzax* environments
Originally, there are two JAX environments (state externally managed) : the *balls environments* ***BallsEnv*** and ***BallsEnvGoal*** (*goal-conditioned RL*), but ***Balzax*** also proposes ***Gym* wrappers** so as to apply most of *state-of-the-art* algorithms on it. In the implemented *Gym* wrappers, the environments vectorization is managed internally. *Balzax* also allows to implement custom environments thanks to the *abstract classes* ***BalzaxEnv*** and ***BalzaxGoalEnv***. 

### Balls environments 

The environment is a window in which there are several balls (disks). One ball can be controlled by an agent while the others only move when they get in touch with the first ball. Basically, an agent has to move the other balls to a desired configuration, defined in a goal space in the case of goal-conditioned RL. 

The ***observations*** can either be the balls *positions* or an *image* of the scene. The ***action*** corresponds to a *speed command*. If it exist, the ***achieved goal*** is calculated from the observation thanks to a *mapping from the observation space to the goal space*. The ***reward*** and ***success*** functions can also be defined externally.

For each step, the ***dynamics of the environment*** follow the substeps below. The agent controls directly the speed of the first ball and the collisions are sequentially managed and directly modify the objects positions.

- ***Agent control :*** $$x_{t+1, \text{agent}}' = x_{t, \text{agent}} + K . g(a_t)$$ Where $a_t$ is the action at timestep $t$, $x_{t, \text{agent}}$ the position of the agent at time $t$, $x_{t+1, \text{agent}}'$ the next position if there is no collision after, $K$ is a constant gain, $g$ is the function transforming the action in a velocity. In the current version, $g$ forces the speed to stay in the unit circle : $$g(v) = \frac{v}{1 + \text{relu}\left(||v|| - 1\right)}$$
- ***Ball-ball collision :*** For $0 \le i \le N_{\text{balls}}-1$ and $i+1 \le j \le N_{\text{balls}}-1$ : $$p_{ij} \leftarrow \text{relu}\left(r_i + r_j - ||x_j - x_i||\right) . \frac{x_j - x_i}{||x_j - x_i||}$$ $$x_i \leftarrow x_i - p_{ij}/2$$ $$x_j \leftarrow x_j + p_{ij}/2$$
- ***Ball-wall collision :*** *The balls are forced to stay in the window* by a **continuous and piecewise linear function** $W$ : $$x \leftarrow W(x)$$

## Examples

### 1. Learning to push a ball to a desired position thanks to SAC and HER

#### Environment (goal-conditioned RL)
- **Two balls :** One *agent*, one *tool* (ball to be pushed)
- **Goal space :** $(x_1,y_1) \in [0, 1]^2$ tool position
- **Observation space :** $(x_0, y_0, x_1,y_1) \in [0, 1]^4$ positions of both
- **Action space :** $(v_x, v_y) \in [-1, 1]^2$ speed command
- **Sparse reward :** $r(s_t^{\text{tool}}, a_t, g) = \left[ || s_t^{\text{tool}} - g || \le \epsilon  \right]$

#### Rendering

After ***1 500 000 learning steps***, the following result is obtained. The balls are in blue while the goal is in red.

https://user-images.githubusercontent.com/91844362/174491520-8bf9e437-6964-4cae-a90a-182c78f81844.mp4


After ***6 000 000 learning steps***:

https://user-images.githubusercontent.com/91844362/175326039-496a98c2-0873-4bd0-adb9-d0293c643ba5.mp4
