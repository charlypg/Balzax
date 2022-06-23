# Balzax
Game environments for reinforcement learning and open-ended learning composed of balls moving other balls and coded in JAX. 

## Installation

### CPU
With conda :
```
conda env create -f environment_cpu.yaml
```

### GPU
*Coming soon...*


## Environments
Originally, there are two JAX environments (state externally managed) : the balls environments ***BallsEnv*** and ***BallsEnvGoal*** (*goal-conditioned RL*), but ***Balzax*** also proposes ***Gym* wrappers** so as to apply most of *state-of-the-art* algorithms on it. In the implemented *Gym* wrappers, the environments vectorization is managed internally. *Balzax* also allows to implement custom environments thanks to the *abstract classes* ***BalzaxEnv*** and ***BalzaxGoalEnv***. 

### Balls environments 

The environment is a window in which there are several balls (disks). One ball can be controlled by an agent while the others only move when they get in touch with the first ball. Basically, an agent has to move the other balls to a desired configuration, defined in a goal space in the case of goal-conditioned RL. 

The ***observations*** can either be the balls *positions* or an *image* of the scene. The ***action*** corresponds to a *speed command*. If it exist, the ***achieved goal*** is calculated from the observation thanks to a *mapping from the observation space to the goal space*. The ***reward*** and ***success*** functions can also be defined externally.

#### Dynamics
*Coming soon*

## Examples

### 1. Learning to push a ball to a desired position thanks to SAC and HER

#### Environment (goal-conditioned RL)
- **Two balls :** One *agent*, one *tool* (ball to be pushed)
- **Goal space :** $(x_1,y_1) \in [0, 1]^2$ tool position
- **Observation space :** $(x_0, y_0, x_1,y_1) \in [0, 1]^4$ positions of both
- **Action space :** $(v_x, v_y) \in [-1, 1]^2$ speed command
- **Sparse reward :** $r(s_t^{\text{tool}}, a_t, g) = \left[ || s_t^{\text{tool}} - g || \le \epsilon  \right]$

#### Rendering

After 1 500 000 learning steps, the following result is obtained : 

https://user-images.githubusercontent.com/91844362/174491520-8bf9e437-6964-4cae-a90a-182c78f81844.mp4
