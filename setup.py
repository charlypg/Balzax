from setuptools import setup, find_packages
from balzax import __version__

setup(
    name="balzax",
    version=__version__,
    author="Charly Pecqueux--Guezenec",
    description="""Balzax is an open source Python library for developing simple,
    vectorized and fast reinforcement learning environments thanks to JAX. It provides
    an API and already implemented game environments to test and compare reinforcement
    learning algorithms.""",
    url="https://github.com/charlypg/Balzax",
    packages=find_packages(),
    install_requires=[],
    license="LICENSE",
)
