"""
Test that the package can be imported correctly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_package():
    """Test importing main package."""
    import swmmEnv

    assert hasattr(swmmEnv, '__version__')
    assert swmmEnv.__version__ == '0.1.0'


def test_import_sim_modules():
    """Test importing simulation modules."""
    from swmmEnv.sim import SWMMEngine, TimeSync, StateNormalizer, MappingRegistry

    assert SWMMEngine is not None
    assert TimeSync is not None
    assert StateNormalizer is not None
    assert MappingRegistry is not None


def test_import_env_modules():
    """Test importing environment modules."""
    from swmmEnv.envs import SWMMEnv, SWMMParallelEnv

    assert SWMMEnv is not None
    assert SWMMParallelEnv is not None


def test_import_config_loader():
    """Test importing config loader."""
    from swmmEnv.config import load_config, validate_config

    assert load_config is not None
    assert validate_config is not None


def test_import_reward_functions():
    """Test importing reward functions."""
    from swmmEnv.reward import default_reward

    assert default_reward is not None


def test_load_default_config():
    """Test loading default configuration."""
    from swmmEnv.config import load_config

    config = load_config(None, merge_defaults=False)

    assert isinstance(config, dict)
    assert 'agents' in config
    assert 'time_sync' in config
    assert 'normalization' in config


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])