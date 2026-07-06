"""
Unit tests for SWMMParallelEnv (PettingZoo wrapper).
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swmmEnv.envs.swmm_env.pettingzoo_env import SWMMParallelEnv


class TestSWMMParallelEnv:
    """Test SWMMParallelEnv class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'inp_file': 'test.inp',
            'agents': {
                'pump_1': {
                    'type': 'pump',
                    'link_id': 'P1',
                    'upstream_node': 'J1',
                    'downstream_node': 'J2'
                },
                'gate_1': {
                    'type': 'weir',
                    'link_id': 'W1',
                    'upstream_node': 'J3',
                    'downstream_node': 'J4'
                }
            },
            'time_sync': {
                'decision_interval': 300,
                'swmm_step': 10
            },
            'normalization': {
                'obs': {
                    'depth': {'mean': 2.0, 'std': 1.5},
                    'flow': {'mean': 0.5, 'std': 0.3},
                    'rainfall': {'mean': 5.0, 'std': 10.0},
                    'setting': {'mean': 0.5, 'std': 0.3}
                },
                'reward': {'mean': 0.0, 'std': 10.0}
            },
            'obs_nodes': ['J1', 'J2', 'J3', 'J4'],
            'max_steps': 100,
            'reward_fn': 'default_reward'
        }

    @pytest.fixture
    @patch('swmmEnv.envs.swmm_env.pettingzoo_env.SWMMEnv')
    def env(self, mock_core_env_class, config):
        """Create environment with mocked core env."""
        mock_core_env = Mock()

        # Mock core env methods
        mock_core_env.reset.return_value = {
            'pump_1': np.zeros(5, dtype=np.float32),
            'gate_1': np.zeros(4, dtype=np.float32)
        }
        mock_core_env.step.return_value = (
            {
                'pump_1': np.zeros(5, dtype=np.float32),
                'gate_1': np.zeros(4, dtype=np.float32)
            },
            -1.0,  # reward
            False,  # done
            {'step_count': 1}  # info
        )
        mock_core_env.get_observation.side_effect = lambda agent: \
            np.zeros(5 if agent == 'pump_1' else 4, dtype=np.float32)
        mock_core_env.close.return_value = None
        mock_core_env.render.return_value = None

        # Mock mapping
        mock_mapping = Mock()
        mock_mapping.get_element_type.side_effect = lambda agent: \
            'pump' if agent == 'pump_1' else 'weir'
        mock_core_env.mapping = mock_mapping

        # Mock agents list
        mock_core_env.agents = ['pump_1', 'gate_1']
        mock_core_env.possible_agents = ['pump_1', 'gate_1']

        mock_core_env_class.return_value = mock_core_env

        env = SWMMParallelEnv(config)

        return env

    def test_init(self, env, config):
        """Test initialization."""
        assert env.config == config
        assert env.possible_agents == ['pump_1', 'gate_1']
        assert env.agents == ['pump_1', 'gate_1']

    def test_observation_spaces(self, env):
        """Test observation spaces are defined."""
        assert 'pump_1' in env.observation_spaces
        assert 'gate_1' in env.observation_spaces

        # Pump has 5-dim observation
        assert env.observation_spaces['pump_1'].shape == (5,)

        # Gate has 4-dim observation
        assert env.observation_spaces['gate_1'].shape == (4,)

    def test_action_spaces(self, env):
        """Test action spaces are defined."""
        assert 'pump_1' in env.action_spaces
        assert 'gate_1' in env.action_spaces

        # Action space should be Box(0, 1, shape=(1,))
        assert env.action_spaces['pump_1'].shape == (1,)
        assert env.action_spaces['pump_1'].low[0] == 0.0
        assert env.action_spaces['pump_1'].high[0] == 1.0

    def test_reset(self, env):
        """Test reset method."""
        observations, infos = env.reset()

        assert isinstance(observations, dict)
        assert isinstance(infos, dict)

        assert 'pump_1' in observations
        assert 'gate_1' in observations

        assert 'pump_1' in infos
        assert 'gate_1' in infos

    def test_step(self, env):
        """Test step method."""
        env.reset()

        actions = {
            'pump_1': np.array([0.8], dtype=np.float32),
            'gate_1': np.array([0.5], dtype=np.float32)
        }

        obs, rewards, terms, truncs, infos = env.step(actions)

        # Check return types
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terms, dict)
        assert isinstance(truncs, dict)
        assert isinstance(infos, dict)

        # Check keys
        assert 'pump_1' in obs
        assert 'gate_1' in obs

        # Rewards should be shared (global reward)
        assert rewards['pump_1'] == rewards['gate_1']

        # Truncations should be False
        assert all(truncs.values()) == False

    def test_step_without_reset_raises(self, env):
        """Test stepping without reset raises error."""
        actions = {
            'pump_1': np.array([0.5], dtype=np.float32),
            'gate_1': np.array([0.5], dtype=np.float32)
        }

        with pytest.raises(RuntimeError, match="must be reset"):
            env.step(actions)

    def test_observe(self, env):
        """Test observe method."""
        env.reset()

        obs = env.observe('pump_1')

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)

    def test_observe_terminated_agent(self, env):
        """Test observing terminated agent returns zeros."""
        env.reset()
        env.agents = []  # Simulate termination

        obs = env.observe('pump_1')

        assert isinstance(obs, np.ndarray)
        assert np.all(obs == 0.0)

    def test_state(self, env):
        """Test state method for centralized critic."""
        env.reset()

        state = env.state()

        # State should concatenate all observations
        # 5 (pump) + 4 (gate) = 9
        assert isinstance(state, np.ndarray)
        assert state.shape == (9,)

    def test_close(self, env):
        """Test close method."""
        env.close()

        env.core_env.close.assert_called_once()
        assert env.agents == []

    def test_render(self, env):
        """Test render method."""
        env.reset()
        env.render()

        env.core_env.render.assert_called()

    def test_get_env_info(self, env):
        """Test get_env_info for MARLlib."""
        info = env.get_env_info()

        assert 'space_obs' in info
        assert 'space_act' in info
        assert 'num_agents' in info
        assert 'episode_limit' in info
        assert 'policy_mapping_info' in info

        assert info['num_agents'] == 2
        assert info['episode_limit'] == 100

    def test_num_agents(self, env):
        """Test num_agents property."""
        assert env.num_agents == 2

    def test_termination_clears_agents(self, env):
        """Test that termination clears agents list."""
        env.reset()

        # Mock step returning done=True
        env.core_env.step.return_value = (
            {'pump_1': np.zeros(5), 'gate_1': np.zeros(4)},
            -1.0,
            True,  # done
            {}
        )

        actions = {
            'pump_1': np.array([0.5], dtype=np.float32),
            'gate_1': np.array([0.5], dtype=np.float32)
        }

        obs, rewards, terms, truncs, infos = env.step(actions)

        # Agents list should be cleared
        assert env.agents == []
        assert terms['pump_1'] == True
        assert terms['gate_1'] == True

    def test_repr(self, env):
        """Test string representation."""
        repr_str = repr(env)

        assert 'SWMMParallelEnv' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])