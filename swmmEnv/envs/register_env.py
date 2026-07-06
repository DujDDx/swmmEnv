"""
MARLlib environment registration for SWMMEnv.

This module provides functions to register SWMMEnv with MARLlib
without modifying MARLlib source code (external registration).

IMPORTANT: Configuration handling is careful not to merge default config
with user config to avoid missing nodes/elements errors.
"""

import os
from typing import Dict, Any, Optional

from swmmEnv.config.loader import load_config, expand_relative_paths, validate_config
from swmmEnv.envs.swmm_env.pettingzoo_env import SWMMParallelEnv


def make_env(
    environment_name: str = "swmm",
    map_name: str = "control",
    config_path: Optional[str] = None,
    worker_index: int = 0,
    **kwargs
) -> SWMMParallelEnv:
    """
    Create SWMM environment for MARLlib/RLlib.

    This function follows MARLlib's environment creation pattern,
    allowing seamless integration with MARLlib training scripts.

    Args:
        environment_name: Environment name (should be "swmm")
        map_name: Scenario/map name (used to find config file)
        config_path: Direct path to config file (overrides map_name lookup)
        worker_index: Worker index for parallel environments (default 0)
        **kwargs: Additional configuration overrides (applied AFTER loading config)

    Returns:
        SWMMParallelEnv instance

    CONFIGURATION PRIORITY:
    1. config_path if provided (NO merging with defaults)
    2. map_name-based lookup (NO merging with defaults)
    3. kwargs overrides (applied to loaded config)

    Example:
        >>> # Basic usage
        >>> env = make_env(map_name="control")
        >>>
        >>> # With custom config (no default merge)
        >>> env = make_env(config_path="configs/my_scenario.yaml")
        >>>
        >>> # For RLlib parallel workers
        >>> env = make_env(config_path="configs/my_scenario.yaml", worker_index=1)
    """
    # Load configuration WITHOUT merging defaults
    # This prevents missing node/element errors from default config
    config = None

    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                f"Please provide a valid configuration file."
            )
        # Load without merging defaults
        config = load_config(config_path, merge_defaults=False)

    else:
        # Try to find config by map_name with flexible path search
        config = _find_config_by_map_name(map_name)

        if config is None:
            raise ValueError(
                f"Configuration not found for map_name '{map_name}'. "
                f"Please either:\n"
                f"  1. Provide a valid config_path\n"
                f"  2. Create configs/{map_name}.yaml\n"
                f"  3. Pass full configuration via kwargs"
            )

    # Apply kwargs overrides (after loading, not merging)
    # This allows user to override specific values without losing config
    for key, value in kwargs.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            # Merge nested dicts carefully
            config[key].update(value)
        else:
            config[key] = value

    # Validate config (ensure all required fields present)
    validate_config(config)

    # Expand relative paths based on config location
    if config_path:
        base_dir = os.path.dirname(os.path.abspath(config_path))
        config = expand_relative_paths(config, base_dir)

    # Set worker index for parallel environments
    config['worker_index'] = worker_index

    # Create environment
    env = SWMMParallelEnv(config)

    return env


def _find_config_by_map_name(map_name: str) -> Optional[Dict[str, Any]]:
    """
    Find configuration file by map_name with flexible path search.

    Searches in multiple standard locations:
    - configs/{map_name}.yaml
    - config/{map_name}.yaml
    - {cwd}/configs/{map_name}.yaml
    - {package}/config/{map_name}.yaml

    Args:
        map_name: Scenario name

    Returns:
        Configuration dict if found, None otherwise
    """
    # Get current working directory
    cwd = os.getcwd()

    # Get package directory
    package_dir = os.path.dirname(os.path.dirname(__file__))

    # Possible locations in priority order
    possible_paths = [
        os.path.join(cwd, f"configs/{map_name}.yaml"),
        os.path.join(cwd, f"config/{map_name}.yaml"),
        os.path.join(cwd, f"{map_name}.yaml"),
        os.path.join(package_dir, f"config/{map_name}.yaml"),
        f"configs/{map_name}.yaml",
        f"config/{map_name}.yaml",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                config = load_config(path, merge_defaults=False)
                print(f"Loaded config from: {path}")
                return config
            except Exception as e:
                print(f"Warning: Failed to load config from {path}: {e}")
                continue

    return None


def register_with_marllib(config_dir: Optional[str] = None) -> None:
    """
    Register SWMMEnv with MARLlib's environment registry.

    This is an optional function for tighter MARLlib integration.
    It attempts to register the environment with MARLlib if installed.

    Args:
        config_dir: Directory containing MARLlib config files

    Note:
        This function modifies MARLlib's ENV_REGISTRY in-memory.
        It does not modify MARLlib source files.
    """
    try:
        import marllib
        from marllib.envs.base_env import ENV_REGISTRY

        # Create environment creator with worker support
        def env_creator(env_config):
            map_name = env_config.get('map_name', 'control')
            config_path = env_config.get('config_path', None)
            worker_index = env_config.get('worker_index', 0)

            return make_env(
                environment_name="swmm",
                map_name=map_name,
                config_path=config_path,
                worker_index=worker_index,
                **env_config
            )

        # Register in MARLlib's registry
        ENV_REGISTRY["swmm"] = env_creator

        print("SWMMEnv registered with MARLlib as 'swmm'")

    except ImportError:
        print(
            "MARLlib not installed. "
            "Install with: pip install marllib"
        )
    except Exception as e:
        print(f"Failed to register with MARLlib: {e}")


def get_marllib_config(map_name: str = "control") -> Dict[str, Any]:
    """
    Generate MARLlib-compatible configuration for SWMMEnv.

    Returns a configuration dictionary that can be used with MARLlib's
    environment creation.

    Args:
        map_name: Scenario name

    Returns:
        MARLlib configuration dictionary
    """
    return {
        "env": "swmm",
        "env_args": {
            "map_name": map_name,
            "config_path": f"configs/{map_name}.yaml",
        },
        "mask_flag": False,
        "global_state_flag": True,  # For centralized critic (MAPPO)
        "opp_action_in_cc": True,
    }


# Convenience function for testing
def create_test_env() -> SWMMParallelEnv:
    """
    Create a test environment with default configuration.

    This is useful for quick testing and debugging.

    Returns:
        SWMMParallelEnv instance with default config
    """
    config = load_config(None)

    # Set some reasonable test values
    config['max_steps'] = 10

    return SWMMParallelEnv(config)


__all__ = [
    'make_env',
    'register_with_marllib',
    'get_marllib_config',
    'create_test_env',
]