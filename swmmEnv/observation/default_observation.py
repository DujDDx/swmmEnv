"""
Default observation functions and builders for SWMMEnv.

This module provides:
- DEFAULT_FEATURES: Default feature configuration (backward compatible)
- OBSERVATION_REGISTRY: Registry for custom observation functions
- build_observation_fn(): Factory for creating observation functions
- compute_obs_dims(): Compute observation dimensions from feature config
"""

from typing import Dict, Any, List, Callable, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from swmmEnv.sim.engine import SWMMEngine
    from swmmEnv.sim.mapping import MappingRegistry
    from swmmEnv.sim.normalizer import StateNormalizer

from swmmEnv.observation.feature_extractors import get_feature_extractor


# Default features (backward compatible with current hard-coded logic)
DEFAULT_FEATURES: Dict[str, List[str]] = {
    'pump': ['upstream_depth', 'downstream_depth', 'flow', 'setting', 'rainfall'],
    'gate': ['upstream_depth', 'downstream_depth', 'setting', 'rainfall'],
    'weir': ['upstream_depth', 'downstream_depth', 'setting', 'rainfall'],
}

# Observation function registry (mirrors REWARD_REGISTRY pattern)
OBSERVATION_REGISTRY: Dict[str, Callable] = {}


def register_observation_fn(name: str, fn: Callable) -> None:
    """
    Register a custom observation function.

    Args:
        name: Function name
        fn: Observation function with signature
            (engine, agent_id, config) -> np.ndarray

    Example:
        >>> def my_obs(engine, agent_id, config):
        ...     return np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> register_observation_fn('my_obs', my_obs)
    """
    if name in OBSERVATION_REGISTRY:
        raise ValueError(f"Observation function '{name}' already registered")
    OBSERVATION_REGISTRY[name] = fn


def get_observation_fn(name: str) -> Callable:
    """
    Get observation function by name.

    Args:
        name: Function name

    Returns:
        Observation function callable

    Raises:
        KeyError: If function not found
    """
    if name not in OBSERVATION_REGISTRY:
        available = list(OBSERVATION_REGISTRY.keys())
        raise KeyError(
            f"Unknown observation function '{name}'. Available: {available}"
        )
    return OBSERVATION_REGISTRY[name]


def compute_obs_dims(features_config: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Compute observation dimensions from feature configuration.

    Args:
        features_config: Dict mapping agent_type to list of feature names

    Returns:
        Dict mapping agent_type to observation dimension

    Example:
        >>> features = {'pump': ['flow', 'setting'], 'gate': ['setting']}
        >>> compute_obs_dims(features)
        {'pump': 2, 'gate': 1}
    """
    return {
        agent_type: len(features)
        for agent_type, features in features_config.items()
    }


def _make_legacy_observation(
    engine: 'SWMMEngine',
    mapping: 'MappingRegistry',
    normalizer: 'StateNormalizer',
    obs_raingage: Optional[str]
) -> Callable:
    """
    Create legacy (hard-coded) observation function.

    This replicates the original get_observation logic for backward compatibility.

    Args:
        engine: SWMMEngine instance
        mapping: MappingRegistry instance
        normalizer: StateNormalizer instance
        obs_raingage: Raingage ID for rainfall observation

    Returns:
        Observation function (agent_id -> np.ndarray)
    """

    def get_observation(agent_id: str) -> np.ndarray:
        agent_config = mapping.get_agent_config(agent_id)
        agent_type = agent_config['type']

        # Get rainfall observation
        rainfall = engine.get_rainfall(obs_raingage)

        if agent_type == 'pump':
            # Pump observation: upstream_depth, downstream_depth, flow, setting, rainfall
            upstream_node = agent_config.get('upstream_node')
            downstream_node = agent_config.get('downstream_node')
            link_id = agent_config['link_id']

            upstream_depth = (
                engine.get_node_state(upstream_node)['depth']
                if upstream_node else 0.0
            )
            downstream_depth = (
                engine.get_node_state(downstream_node)['depth']
                if downstream_node else 0.0
            )
            link_state = engine.get_link_state(link_id)
            flow = link_state['flow']
            setting = link_state['current_setting']

            raw_obs = np.array([
                upstream_depth,
                downstream_depth,
                flow,
                setting,
                rainfall
            ], dtype=np.float32)

            # Normalize
            obs_names = ['depth', 'depth', 'flow', 'setting', 'rainfall']
            normalized_obs = normalizer.normalize_obs(raw_obs, obs_names)

        else:  # gate or weir
            # Observation: upstream_depth, downstream_depth, setting, rainfall
            upstream_node = agent_config.get('upstream_node')
            downstream_node = agent_config.get('downstream_node')
            link_id = agent_config['link_id']

            upstream_depth = (
                engine.get_node_state(upstream_node)['depth']
                if upstream_node else 0.0
            )
            downstream_depth = (
                engine.get_node_state(downstream_node)['depth']
                if downstream_node else 0.0
            )
            link_state = engine.get_link_state(link_id)
            setting = link_state['current_setting']

            raw_obs = np.array([
                upstream_depth,
                downstream_depth,
                setting,
                rainfall
            ], dtype=np.float32)

            # Normalize
            obs_names = ['depth', 'depth', 'setting', 'rainfall']
            normalized_obs = normalizer.normalize_obs(raw_obs, obs_names)

        return normalized_obs

    return get_observation


def _make_declarative_observation(
    engine: 'SWMMEngine',
    mapping: 'MappingRegistry',
    normalizer: 'StateNormalizer',
    obs_raingage: Optional[str],
    features_config: Dict[str, List[str]]
) -> Callable:
    """
    Create declarative (config-driven) observation function.

    Args:
        engine: SWMMEngine instance
        mapping: MappingRegistry instance
        normalizer: StateNormalizer instance
        obs_raingage: Raingage ID for rainfall observation
        features_config: Dict mapping agent_type to list of feature names

    Returns:
        Observation function (agent_id -> np.ndarray)
    """

    def get_observation(agent_id: str) -> np.ndarray:
        agent_config = mapping.get_agent_config(agent_id)
        agent_type = agent_config['type']

        # Get feature list for this agent type
        feature_names = features_config.get(agent_type, features_config.get('default', []))

        if not feature_names:
            raise ValueError(
                f"No features defined for agent type '{agent_type}' "
                f"and no 'default' features specified"
            )

        # Extract raw values
        raw_values = []
        norm_names = []

        for fname in feature_names:
            extractor_fn, norm_name = get_feature_extractor(fname)
            value = extractor_fn(engine, agent_config, obs_raingage)
            raw_values.append(value)
            norm_names.append(norm_name)

        # Build and normalize observation
        raw_obs = np.array(raw_values, dtype=np.float32)
        normalized_obs = normalizer.normalize_obs(raw_obs, norm_names)

        return normalized_obs

    return get_observation


def build_observation_fn(
    config: Dict[str, Any],
    engine: 'SWMMEngine',
    mapping: 'MappingRegistry',
    normalizer: 'StateNormalizer',
    obs_raingage: Optional[str]
) -> Callable:
    """
    Build observation function from configuration.

    This is the main factory function that inspects the config and
    returns the appropriate observation function.

    Three modes:
    - No 'observation' key: return legacy hard-coded observation
    - mode == 'declarative': return config-driven observation
    - mode == 'custom': return user-injected function

    Args:
        config: Configuration dictionary
        engine: SWMMEngine instance
        mapping: MappingRegistry instance
        normalizer: StateNormalizer instance
        obs_raingage: Raingage ID for rainfall observation

    Returns:
        Observation function with signature (agent_id: str) -> np.ndarray
    """
    obs_cfg = config.get('observation', None)

    # No observation config: use legacy hard-coded logic (backward compatible)
    if obs_cfg is None:
        return _make_legacy_observation(engine, mapping, normalizer, obs_raingage)

    mode = obs_cfg.get('mode', 'declarative')

    if mode == 'declarative':
        # Config-driven observation from feature lists
        features_config = obs_cfg.get('features', DEFAULT_FEATURES)
        return _make_declarative_observation(
            engine, mapping, normalizer, obs_raingage, features_config
        )

    elif mode == 'custom':
        # User-injected observation function
        fn_spec = obs_cfg.get('observation_fn')

        if fn_spec is None:
            raise ValueError(
                "observation.mode == 'custom' requires 'observation_fn' "
                "to be specified (string or callable)"
            )

        # Resolve function
        if isinstance(fn_spec, str):
            fn = get_observation_fn(fn_spec)
        elif callable(fn_spec):
            fn = fn_spec
        else:
            raise ValueError(
                f"observation_fn must be a string (registry name) or callable, "
                f"got {type(fn_spec)}"
            )

        # Wrap to match (agent_id) signature
        def get_observation(agent_id: str) -> np.ndarray:
            return fn(engine, agent_id, config)

        return get_observation

    else:
        raise ValueError(
            f"Unknown observation mode '{mode}'. "
            f"Must be 'declarative' or 'custom'"
        )


__all__ = [
    'DEFAULT_FEATURES',
    'OBSERVATION_REGISTRY',
    'register_observation_fn',
    'get_observation_fn',
    'compute_obs_dims',
    'build_observation_fn',
]