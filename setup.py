from setuptools import setup, find_packages
from balzax import __version__

setup(
    name="balzax",
    version=__version__,
    author="Charly Pecqueux--Guezenec",
    description="""Game environments for reinforcement learning and open-ended learning
composed of balls moving other balls and coded in JAX.""",
    url="https://github.com/charlypg/Balzax",
    packages=find_packages(),
    install_requires=[],
    license="LICENSE",
)
