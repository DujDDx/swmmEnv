"""
Unit tests for observation configuration system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from swmmEnv.config.loader import validate_config


class TestFeatureExtractors:
    """Test feature extractor registry and functions."""

    def test_registry_has_required_features(self):
        """Test that required features are registered."""
        required = ['upstream_depth', 'downstream_depth', 'flow', 'setting', 'rainfall']
        for feature in required:
            assert feature in FEATURE_EXTRACTOR_REGISTRY

    def test_get_feature_extractor(self):
        """Test getting feature extractor."""
        extractor_fn, norm_name = get_feature_extractor('flow')
        assert callable(extractor_fn)
        assert norm_name == 'flow'

    def test_get_feature_extractor_unknown(self):
        """Test error for unknown feature."""
        with pytest.raises(KeyError, match="Unknown feature"):
            get_feature_extractor('nonexistent_feature')

    def test_register_feature_extractor(self):
        """Test registering custom feature extractor."""
        def my_extractor(engine, agent_config, obs_raingage):
            return 42.0

        register_feature_extractor('test_feature', my_extractor, 'test')
        assert 'test_feature' in FEATURE_EXTRACTOR_REGISTRY

        extractor_fn, norm_name = get_feature_extractor('test_feature')
        assert extractor_fn is my_extractor
        assert norm_name == 'test'

    def test_register_duplicate_feature(self):
        """Test error when registering duplicate feature."""
        def my_extractor(engine, agent_config, obs_raingage):
            return 0.0

        # First registration should work
        register_feature_extractor('dup_test', my_extractor, 'test')

        # Second should fail
        with pytest.raises(ValueError, match="already registered"):
            register_feature_extractor('dup_test', my_extractor, 'test')

    def test_list_available_features(self):
        """Test listing available features."""
        features = list_available_features()
        assert isinstance(features, list)
        assert 'upstream_depth' in features
        assert 'flow' in features


class TestComputeObsDims:
    """Test observation dimension computation."""

    def test_compute_obs_dims_default(self):
        """Test default features dimension."""
        dims = compute_obs_dims(DEFAULT_FEATURES)
        assert dims['pump'] == 5
        assert dims['gate'] == 4
        assert dims['weir'] == 4

    def test_compute_obs_dims_custom(self):
        """Test custom features dimension."""
        features = {
            'pump': ['flow', 'setting'],
            'gate': ['setting'],
        }
        dims = compute_obs_dims(features)
        assert dims['pump'] == 2
        assert dims['gate'] == 1


class TestObservationRegistry:
    """Test observation function registry."""

    def test_register_observation_fn(self):
        """Test registering observation function."""
        def my_obs(engine, agent_id, config):
            return np.array([1.0, 2.0])

        register_observation_fn('test_obs', my_obs)
        assert 'test_obs' in OBSERVATION_REGISTRY

        fn = get_observation_fn('test_obs')
        assert fn is my_obs

    def test_get_observation_fn_unknown(self):
        """Test error for unknown observation function."""
        with pytest.raises(KeyError, match="Unknown observation function"):
            get_observation_fn('nonexistent_fn')


class TestBuildObservationFn:
    """Test observation function builder."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        engine = Mock()
        engine.get_node_state.return_value = {
            'depth': 2.0, 'head': 10.0, 'volume': 100.0,
            'flooding': 0.0, 'total_inflow': 0.5
        }
        engine.get_link_state.return_value = {
            'flow': 0.5, 'depth': 0.5, 'volume': 10.0, 'current_setting': 0.5
        }
        engine.get_rainfall.return_value = 5.0
        return engine

    @pytest.fixture
    def mock_mapping(self):
        """Create mock mapping."""
        mapping = Mock()
        mapping.get_agent_config.return_value = {
            'type': 'pump',
            'link_id': 'P1',
            'upstream_node': 'J1',
            'downstream_node': 'J2'
        }
        return mapping

    @pytest.fixture
    def mock_normalizer(self):
        """Create mock normalizer."""
        normalizer = Mock()
        normalizer.normalize_obs.side_effect = lambda obs, names: obs
        return normalizer

    def test_build_legacy_observation(self, mock_engine, mock_mapping, mock_normalizer):
        """Test legacy observation function (no config)."""
        config = {}  # No observation key

        obs_fn = build_observation_fn(
            config, mock_engine, mock_mapping, mock_normalizer, 'RG1'
        )

        obs = obs_fn('pump_1')
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)  # Default pump obs dim

    def test_build_declarative_observation(self, mock_engine, mock_mapping, mock_normalizer):
        """Test declarative observation function."""
        config = {
            'observation': {
                'mode': 'declarative',
                'features': {
                    'pump': ['flow', 'setting', 'rainfall']
                }
            }
        }

        obs_fn = build_observation_fn(
            config, mock_engine, mock_mapping, mock_normalizer, 'RG1'
        )

        obs = obs_fn('pump_1')
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3,)  # 3 features specified

    def test_build_custom_observation_callable(self, mock_engine, mock_mapping, mock_normalizer):
        """Test custom observation function via callable."""
        def my_obs(engine, agent_id, config):
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

        config = {
            'observation': {
                'mode': 'custom',
                'observation_fn': my_obs
            }
        }

        obs_fn = build_observation_fn(
            config, mock_engine, mock_mapping, mock_normalizer, 'RG1'
        )

        obs = obs_fn('pump_1')
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3,)
        np.testing.assert_array_equal(obs, [1.0, 2.0, 3.0])

    def test_build_custom_observation_string(self, mock_engine, mock_mapping, mock_normalizer):
        """Test custom observation function via string lookup."""
        def my_obs(engine, agent_id, config):
            return np.array([4.0, 5.0], dtype=np.float32)

        register_observation_fn('test_string_obs', my_obs)

        config = {
            'observation': {
                'mode': 'custom',
                'observation_fn': 'test_string_obs'
            }
        }

        obs_fn = build_observation_fn(
            config, mock_engine, mock_mapping, mock_normalizer, 'RG1'
        )

        obs = obs_fn('pump_1')
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2,)


class TestCustomObservationFunction:
    """Test CustomObservationFunction base class."""

    def test_custom_observation_subclass(self):
        """Test creating custom observation subclass."""
        class MyObs(CustomObservationFunction):
            def __call__(self, engine, agent_id, config):
                return np.array([1.0, 2.0], dtype=np.float32)

            def get_obs_dim(self):
                return 2

        obs_fn = MyObs()
        assert obs_fn.get_obs_dim() == 2

    def test_base_class_raises(self):
        """Test base class raises NotImplementedError."""
        obs_fn = CustomObservationFunction()

        with pytest.raises(NotImplementedError):
            obs_fn(Mock(), 'agent_1', {})

        with pytest.raises(NotImplementedError):
            obs_fn.get_obs_dim()


class TestHistoryObservationFunction:
    """Test HistoryObservationFunction example."""

    @pytest.fixture
    def mock_engine(self):
        engine = Mock()
        engine.get_node_state.return_value = {
            'depth': 2.0, 'head': 10.0, 'volume': 100.0,
            'flooding': 0.0, 'total_inflow': 0.5
        }
        engine.get_link_state.return_value = {
            'flow': 0.5, 'depth': 0.5, 'volume': 10.0, 'current_setting': 0.5
        }
        engine.get_rainfall.return_value = 5.0
        return engine

    def test_history_observation(self, mock_engine):
        """Test history observation function."""
        config = {
            'agents': {
                'pump_1': {
                    'type': 'pump',
                    'link_id': 'P1',
                    'upstream_node': 'J1',
                    'downstream_node': 'J2'
                }
            }
        }

        obs_fn = HistoryObservationFunction(history_length=2, base_features=['flow', 'setting'])

        # First call: only 1 history entry (2 features)
        obs1 = obs_fn(mock_engine, 'pump_1', config)
        assert obs1.shape == (2,)  # 2 features * 1 history entry

        # Second call: now we have 2 history entries
        obs2 = obs_fn(mock_engine, 'pump_1', config)
        assert obs2.shape == (4,)  # 2 features * 2 history entries

        obs_fn.reset()
        assert obs_fn.history == {}


class TestConfigValidation:
    """Test observation config validation."""

    def test_validate_declarative_config(self):
        """Test validating declarative observation config."""
        config = {
            'agents': {'pump_1': {'type': 'pump', 'link_id': 'P1'}},
            'time_sync': {'decision_interval': 300, 'swmm_step': 10},
            'normalization': {'obs': {}, 'reward': {}},
            'observation': {
                'mode': 'declarative',
                'features': {
                    'pump': ['flow', 'setting']
                }
            }
        }
        # Should not raise
        validate_config(config)

    def test_validate_custom_config_with_callable(self):
        """Test validating custom observation config with callable."""
        config = {
            'agents': {'pump_1': {'type': 'pump', 'link_id': 'P1'}},
            'time_sync': {'decision_interval': 300, 'swmm_step': 10},
            'normalization': {'obs': {}, 'reward': {}},
            'observation': {
                'mode': 'custom',
                'observation_fn': lambda e, a, c: np.array([1.0])
            }
        }
        # Should not raise
        validate_config(config)

    def test_validate_invalid_mode(self):
        """Test validation rejects invalid mode."""
        config = {
            'agents': {'pump_1': {'type': 'pump', 'link_id': 'P1'}},
            'time_sync': {'decision_interval': 300, 'swmm_step': 10},
            'normalization': {'obs': {}, 'reward': {}},
            'observation': {
                'mode': 'invalid_mode'
            }
        }
        with pytest.raises(ValueError, match="must be 'declarative' or 'custom'"):
            validate_config(config)

    def test_validate_unknown_feature(self):
        """Test validation rejects unknown feature."""
        config = {
            'agents': {'pump_1': {'type': 'pump', 'link_id': 'P1'}},
            'time_sync': {'decision_interval': 300, 'swmm_step': 10},
            'normalization': {'obs': {}, 'reward': {}},
            'observation': {
                'mode': 'declarative',
                'features': {
                    'pump': ['unknown_feature']
                }
            }
        }
        with pytest.raises(ValueError, match="Unknown feature"):
            validate_config(config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])