"""
SWMMEnv - Multi-agent reinforcement learning environment for SWMM stormwater simulation.

This package provides PettingZoo- and RLlib-compatible environments for training
multi-agent reinforcement learning algorithms on SWMM (Storm Water Management
Model) simulations.
"""

__version__ = "0.1.0"

from swmmEnv.envs.swmm_env.pettingzoo_env import SWMMParallelEnv
from swmmEnv.envs.swmm_env.env import SWMMEnv
from swmmEnv.envs.swmm_env.rllib_env import SWMMMultiAgentEnv
from swmmEnv.sim.engine import SWMMEngine
from swmmEnv.config.loader import load_config

# Observation module exports
from swmmEnv.observation import (
    DEFAULT_FEATURES,
    OBSERVATION_REGISTRY,
    CustomObservationFunction,
    HistoryObservationFunction,
    get_observation_fn,
    register_observation_fn,
    get_feature_extractor,
    register_feature_extractor,
    list_available_features,
)

__all__ = [
    # Environments
    "SWMMParallelEnv",
    "SWMMMultiAgentEnv",
    "SWMMEnv",
    "SWMMEngine",
    "load_config",
    # Observation module
    "DEFAULT_FEATURES",
    "OBSERVATION_REGISTRY",
    "CustomObservationFunction",
    "HistoryObservationFunction",
    "get_observation_fn",
    "register_observation_fn",
    "get_feature_extractor",
    "register_feature_extractor",
    "list_available_features",
]