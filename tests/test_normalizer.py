"""
Unit tests for StateNormalizer.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swmmEnv.sim.normalizer import StateNormalizer


class TestStateNormalizer:
    """Test StateNormalizer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'obs': {
                'depth': {'mean': 2.0, 'std': 1.5},
                'flow': {'mean': 0.5, 'std': 0.3},
                'rainfall': {'mean': 5.0, 'std': 10.0},
                'setting': {'mean': 0.5, 'std': 0.3}
            },
            'reward': {'mean': 0.0, 'std': 10.0}
        }

    @pytest.fixture
    def normalizer(self, config):
        """Create normalizer instance."""
        return StateNormalizer(config)

    def test_init(self, normalizer, config):
        """Test initialization."""
        assert normalizer.config == config
        assert 'depth' in normalizer.obs_params
        assert 'flow' in normalizer.obs_params

    def test_normalize_obs(self, normalizer):
        """Test observation normalization."""
        obs = np.array([2.0, 0.5, 5.0, 0.5], dtype=np.float32)
        obs_names = ['depth', 'flow', 'rainfall', 'setting']

        normalized = normalizer.normalize_obs(obs, obs_names)

        # All values equal to mean should normalize to ~0
        assert np.allclose(normalized, np.zeros(4), atol=0.1)

    def test_normalize_obs_different_values(self, normalizer):
        """Test normalization with different values."""
        obs = np.array([3.5, 0.8, 15.0, 0.8], dtype=np.float32)
        obs_names = ['depth', 'flow', 'rainfall', 'setting']

        normalized = normalizer.normalize_obs(obs, obs_names)

        # depth: (3.5 - 2.0) / 1.5 = 1.0
        assert normalized[0] == pytest.approx(1.0, abs=0.01)

        # flow: (0.8 - 0.5) / 0.3 = 1.0
        assert normalized[1] == pytest.approx(1.0, abs=0.01)

        # rainfall: (15.0 - 5.0) / 10.0 = 1.0
        assert normalized[2] == pytest.approx(1.0, abs=0.01)

    def test_normalize_obs_value(self, normalizer):
        """Test single value normalization."""
        # depth = 2.0, should normalize to 0
        normalized = normalizer.normalize_obs_value(2.0, 'depth')
        assert normalized == pytest.approx(0.0, abs=0.01)

        # depth = 3.5, should normalize to 1.0
        normalized = normalizer.normalize_obs_value(3.5, 'depth')
        assert normalized == pytest.approx(1.0, abs=0.01)

    def test_normalize_reward(self, normalizer):
        """Test reward normalization."""
        # Reward = 0 should normalize to 0
        normalized = normalizer.normalize_reward(0.0)
        assert normalized == pytest.approx(0.0, abs=0.01)

        # Reward = 10 should normalize to 1.0
        normalized = normalizer.normalize_reward(10.0)
        assert normalized == pytest.approx(1.0, abs=0.01)

    def test_denormalize_reward(self, normalizer):
        """Test reward denormalization."""
        # Normalized 1.0 should denormalize to 10.0
        denormalized = normalizer.denormalize_reward(1.0)
        assert denormalized == pytest.approx(10.0, abs=0.01)

        # Normalized 0.0 should denormalize to 0.0
        denormalized = normalizer.denormalize_reward(0.0)
        assert denormalized == pytest.approx(0.0, abs=0.01)

    def test_min_max_normalize(self, normalizer):
        """Test min-max normalization."""
        # Value 5 in range [0, 10] should normalize to 0.5
        normalized = normalizer.min_max_normalize(5.0, 0.0, 10.0)
        assert normalized == pytest.approx(0.5, abs=0.01)

        # Value 0 in range [0, 10] should normalize to 0.0
        normalized = normalizer.min_max_normalize(0.0, 0.0, 10.0)
        assert normalized == pytest.approx(0.0, abs=0.01)

        # Value 10 in range [0, 10] should normalize to 1.0
        normalized = normalizer.min_max_normalize(10.0, 0.0, 10.0)
        assert normalized == pytest.approx(1.0, abs=0.01)

    def test_update_stats_online(self):
        """Test online statistics update."""
        config = {
            'obs': {'depth': {'mean': 0.0, 'std': 1.0}},
            'update_online': True
        }
        normalizer = StateNormalizer(config)

        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        normalizer.update_stats(obs, ['depth', 'depth', 'depth'])

        stats = normalizer.get_running_stats()
        assert 'depth' in stats

    def test_clip_and_normalize(self, normalizer):
        """Test clip and normalize."""
        # Value outside range should be clipped
        value = 10.0
        clip_range = (0.0, 5.0)

        normalized = normalizer.clip_and_normalize(
            value, 'depth', clip_range
        )

        # Should normalize clipped value (5.0) not original (10.0)
        expected = (5.0 - 2.0) / 1.5  # (clipped - mean) / std
        assert normalized == pytest.approx(expected, abs=0.01)

    def test_repr(self, normalizer):
        """Test string representation."""
        repr_str = repr(normalizer)

        assert 'StateNormalizer' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])