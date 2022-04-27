from setuptools import setup

setup(
      name="balzax",
      version="0.0.1",
      author="Charly Pecqueux--Guezenec",
      description="Game environment for reinforcement learning and open-ended learning composed of balls moving other balls and coded in JAX.",
      url="https://github.com/charlypg/Balzax",
      packages=["balzax"],
      install_requires=[
              "flax>=0.4.1",
              "jax>=0.3.7",
              "jaxlib>=0.3.7",
              "matplotlib>=3.5.1",
              "numpy>=1.22.3"
          ],
      license="LICENSE"
)