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

https://user-images.githubusercontent.com/91844362/174491520-8bf9e437-6964-4cae-a90a-182c78f81844.mp4

**Environment :**
- Two balls : One *agent*, one *tool* (ball to be pushed)
- Goal space : $[0, 1]^2$ tool position
- Observation space :
