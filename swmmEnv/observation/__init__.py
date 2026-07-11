"""
Observation configuration module for SWMMEnv.

This module provides:
- Feature extractor registry for declarative observation config
- Observation function registry for custom function injection
- Factory function to build observation functions from config
"""

from swmmEnv.observation.feature_extractors import (
    FEATURE_EXTRACTOR_REGISTRY,
    get_feature_extractor,
    register_feature_extractor,
    list_available_features,
)
from swmmEnv.observation.default_observation import (
    DEFAULT_FEATURES,
    OBSERVATION_REGISTRY,
    build_observation_fn,
    compute_obs_dims,
    get_observation_fn,
    register_observation_fn,
)
from swmmEnv.observation.custom_observation import (
    CustomObservationFunction,
    HistoryObservationFunction,
)

__all__ = [
    # Feature extractors
    'FEATURE_EXTRACTOR_REGISTRY',
    'get_feature_extractor',
    'register_feature_extractor',
    'list_available_features',
    # Default observation
    'DEFAULT_FEATURES',
    'OBSERVATION_REGISTRY',
    'build_observation_fn',
    'compute_obs_dims',
    'get_observation_fn',
    'register_observation_fn',
    # Custom observation
    'CustomObservationFunction',
    'HistoryObservationFunction',
]