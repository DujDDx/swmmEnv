"""
Setup script for swmmEnv.

This is a fallback for older pip versions that don't support PEP 660.
For modern pip versions, use pyproject.toml instead.
"""

from setuptools import setup, find_packages

setup(
    name="swmmEnv",
    version="0.1.0",
    description="Multi-agent reinforcement learning environment for SWMM stormwater simulation",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/swmmEnv",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pyswmm>=2.1.0",
        "pettingzoo>=1.24.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
        "marl": [
            "marllib",
            "ray[rllib]>=2.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)