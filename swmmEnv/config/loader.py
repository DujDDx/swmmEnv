"""
Configuration loader for SWMMEnv.

Handles loading, validation, and merging of YAML configuration files.
"""

import os
from typing import Dict, Any, Optional
import yaml

# Default configuration path
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    'default_config.yaml'
)


def load_config(
    config_path: Optional[str] = None,
    merge_defaults: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.
                    If None, returns default configuration.
        merge_defaults: If True, merge with default configuration.

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config_path doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if config_path is None:
        with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if merge_defaults:
        default_config = load_config(None, merge_defaults=False)
        config = merge_with_defaults(config, default_config)

    validate_config(config)

    return config


def _validate_action_space(action_space: Dict[str, Any]) -> None:
    """
    Validate action space configuration.

    Args:
        action_space: Action space configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    action_type = action_space.get('type', 'continuous')

    if action_type not in {'continuous', 'discrete'}:
        raise ValueError(
            f"Invalid action_space type '{action_type}'. "
            f"Must be 'continuous' or 'discrete'"
        )

    if action_type == 'continuous':
        low = action_space.get('low', 0.0)
        high = action_space.get('high', 1.0)
        if low >= high:
            raise ValueError(
                f"action_space.low ({low}) must be less than action_space.high ({high})"
            )

        shape = action_space.get('shape', [1])
        if not isinstance(shape, list) or any(not isinstance(d, int) or d <= 0 for d in shape):
            raise ValueError(f"action_space.shape must be a list of positive integers, got {shape}")
    else:  # discrete
        n = action_space.get('n', 11)
        if not isinstance(n, int) or n < 2:
            raise ValueError(
                f"action_space.n must be an integer >= 2 for discrete action space, got {n}"
            )


def _validate_observation_config(observation: Dict[str, Any]) -> None:
    """
    Validate observation configuration.

    Args:
        observation: Observation configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    mode = observation.get('mode', 'declarative')

    if mode not in ('declarative', 'custom'):
        raise ValueError(
            f"observation.mode must be 'declarative' or 'custom', got '{mode}'"
        )

    if mode == 'declarative':
        features = observation.get('features')
        if features is None:
            raise ValueError(
                "observation.features is required when mode == 'declarative'"
            )
        if not isinstance(features, dict):
            raise ValueError(
                "observation.features must be a dict mapping agent_type to feature list"
            )
        for agent_type, feature_list in features.items():
            if not isinstance(feature_list, list) or not feature_list:
                raise ValueError(
                    f"observation.features.{agent_type} must be a non-empty list"
                )
            # Validate each feature name is known
            from swmmEnv.observation.feature_extractors import FEATURE_EXTRACTOR_REGISTRY
            for fname in feature_list:
                if fname not in FEATURE_EXTRACTOR_REGISTRY:
                    available = list(FEATURE_EXTRACTOR_REGISTRY.keys())
                    raise ValueError(
                        f"Unknown feature '{fname}' in observation.features.{agent_type}. "
                        f"Available features: {available}"
                    )

    elif mode == 'custom':
        fn = observation.get('observation_fn')
        if fn is None:
            raise ValueError(
                "observation.observation_fn is required when mode == 'custom'"
            )
        if not isinstance(fn, str) and not callable(fn):
            raise ValueError(
                "observation.observation_fn must be a string (registry name) or callable"
            )


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and values.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['agents', 'time_sync', 'normalization']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate agents
    if not config['agents']:
        raise ValueError("At least one agent must be defined")

    for agent_id, agent_config in config['agents'].items():
        if 'type' not in agent_config:
            raise ValueError(f"Agent {agent_id}: missing 'type'")

        if 'link_id' not in agent_config:
            raise ValueError(f"Agent {agent_id}: missing 'link_id'")

        if agent_config['type'] not in {'pump', 'gate', 'weir'}:
            raise ValueError(
                f"Agent {agent_id}: invalid type '{agent_config['type']}'. "
                f"Must be 'pump', 'gate', or 'weir'"
            )

    # Validate time_sync
    time_sync = config['time_sync']
    if 'decision_interval' not in time_sync:
        raise ValueError("Missing 'decision_interval' in time_sync")

    if 'swmm_step' not in time_sync:
        raise ValueError("Missing 'swmm_step' in time_sync")

    decision_interval = time_sync['decision_interval']
    swmm_step = time_sync['swmm_step']

    if decision_interval <= 0 or swmm_step <= 0:
        raise ValueError("time_sync values must be positive")

    if decision_interval % swmm_step != 0:
        raise ValueError(
            f"decision_interval ({decision_interval}) must be "
            f"divisible by swmm_step ({swmm_step})"
        )

    # Validate normalization
    normalization = config['normalization']
    if 'obs' not in normalization:
        raise ValueError("Missing 'obs' in normalization")

    if 'reward' not in normalization:
        raise ValueError("Missing 'reward' in normalization")

    # Validate action_space if provided
    if 'action_space' in config:
        _validate_action_space(config['action_space'])

    # Validate observation config if provided
    if 'observation' in config:
        _validate_observation_config(config['observation'])

    # Validate inp_file if provided
    if 'inp_file' in config:
        inp_file = config['inp_file']
        if not isinstance(inp_file, str):
            raise ValueError("inp_file must be a string path")


def merge_with_defaults(
    config: Dict[str, Any],
    defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge configuration with default values.

    Values in config override defaults. Nested dictionaries are merged recursively.

    Args:
        config: User configuration
        defaults: Default configuration

    Returns:
        Merged configuration
    """
    result = defaults.copy()

    for key, value in config.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            # Recursively merge nested dictionaries
            result[key] = merge_with_defaults(value, result[key])
        else:
            # Override with user value
            result[key] = value

    return result


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return load_config(None, merge_defaults=False)


def expand_relative_paths(
    config: Dict[str, Any],
    base_dir: str
) -> Dict[str, Any]:
    """
    Expand relative paths in configuration to absolute paths.

    Args:
        config: Configuration dictionary
        base_dir: Base directory for relative paths

    Returns:
        Configuration with expanded paths
    """
    path_keys = ['inp_file', 'rain_file', 'hotstart_file']

    for key in path_keys:
        if key in config and config[key] is not None:
            path = config[key]
            if not os.path.isabs(path):
                config[key] = os.path.abspath(os.path.join(base_dir, path))

    return config