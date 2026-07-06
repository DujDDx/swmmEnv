"""
MARLlib environment registration for SWMMEnv.

This module provides functions to register SWMMEnv with MARLlib
without modifying MARLlib source code (external registration).
"""

from typing import Dict, Any, Optional

from swmmEnv.config.loader import load_config, expand_relative_paths
from swmmEnv.envs.swmm_env.pettingzoo_env import SWMMParallelEnv


def make_env(
    environment_name: str = "swmm",
    map_name: str = "control",
    config_path: Optional[str] = None,
    **kwargs
) -> SWMMParallelEnv:
    """
    Create SWMM environment for MARLlib.

    This function follows MARLlib's environment creation pattern,
    allowing seamless integration with MARLlib training scripts.

    Args:
        environment_name: Environment name (should be "swmm")
        map_name: Scenario/map name (used to find config file)
        config_path: Direct path to config file (overrides map_name lookup)
        **kwargs: Additional configuration overrides

    Returns:
        SWMMParallelEnv instance

    Example:
        >>> # Basic usage
        >>> env = make_env(map_name="control")
        >>>
        >>> # With custom config
        >>> env = make_env(config_path="configs/my_scenario.yaml")
        >>>
        >>> # For MARLlib integration
        >>> from marllib import marl
        >>> env = make_env(environment_name="swmm", map_name="control")
    """
    # Load configuration
    if config_path is not None:
        config = load_config(config_path)
    else:
        # Try to find config by map_name
        # Look in standard locations
        import os

        possible_paths = [
            f"configs/{map_name}.yaml",
            f"config/{map_name}.yaml",
            f"swmmEnv/config/{map_name}.yaml",
        ]

        config = None
        for path in possible_paths:
            if os.path.exists(path):
                config = load_config(path)
                break

        if config is None:
            # Use default config
            config = load_config(None)

    # Apply overrides
    config.update(kwargs)

    # Expand relative paths
    if config_path:
        base_dir = config_path.rsplit('/', 1)[0]
        config = expand_relative_paths(config, base_dir)

    # Create environment
    env = SWMMParallelEnv(config)

    return env


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

        # Create environment creator
        def env_creator(env_config):
            map_name = env_config.get('map_name', 'control')
            config_path = env_config.get('config_path', None)
            return make_env(
                environment_name="swmm",
                map_name=map_name,
                config_path=config_path,
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