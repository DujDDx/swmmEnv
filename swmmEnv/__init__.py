"""
SWMMEnv - Multi-agent reinforcement learning environment for SWMM stormwater simulation.

This package provides a PettingZoo-compatible environment for training multi-agent
reinforcement learning algorithms on SWMM (Storm Water Management Model) simulations.
"""

__version__ = "0.1.0"

from swmmEnv.envs.swmm_env.pettingzoo_env import SWMMParallelEnv
from swmmEnv.envs.swmm_env.env import SWMMEnv
from swmmEnv.sim.engine import SWMMEngine
from swmmEnv.config.loader import load_config

__all__ = [
    "SWMMParallelEnv",
    "SWMMEnv",
    "SWMMEngine",
    "load_config",
]