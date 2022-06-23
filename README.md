# Balzax
Game environment for reinforcement learning and open-ended learning composed of balls moving other balls and coded in JAX. 

## Installation

### CPU

With conda :

```
conda env create -f environment_cpu.yaml
```

### GPU

*Coming soon...*

## Environments

*Coming soon...*

## Examples

### 1. Learning to push a ball to a desired position thanks to SAC and HER

#### Environment (goal-conditioned)
- **Two balls :** One *agent*, one *tool* (ball to be pushed)
- **Goal space :** $(x_1,y_1) \in [0, 1]^2$ tool position
- **Observation space :** $(x_0, y_0, x_1,y_1) \in [0, 1]^4$ positions of both
- **Action space :** $(v_x, v_y) \in [-1, 1]^2$ speed command
- **Sparse reward :** $r(s_t^{\text{tool}}, a_t, g) = \left[ || s_t^{\text{tool}} - g || \le \epsilon  \right]$

#### Rendering

After 1 500 000 learning steps, the following result is obtained : 

https://user-images.githubusercontent.com/91844362/174491520-8bf9e437-6964-4cae-a90a-182c78f81844.mp4
