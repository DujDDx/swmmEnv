"""
Unit tests for SWMMEnv core MDP.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swmmEnv.envs.swmm_env.env import SWMMEnv


class TestSWMMEnv:
    """Test SWMMEnv class."""

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
            'obs_raingage': 'RG1',
            'max_steps': 100,
            'reward_fn': 'default_reward'
        }

    @pytest.fixture
    @patch('swmmEnv.envs.swmm_env.env.SWMMEngine')
    def env(self, mock_engine_class, config):
        """Create environment instance with mocked engine."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        # Mock engine methods
        mock_engine.reset.return_value = None
        mock_engine.step.return_value = None
        mock_engine.is_ended.return_value = False
        mock_engine.get_current_time.return_value = '2024-01-01 00:00:00'
        mock_engine.get_rainfall.return_value = 5.0
        mock_engine.get_total_flooding.return_value = 0.0
        mock_engine.get_node_state.return_value = {
            'depth': 2.0,
            'head': 10.0,
            'volume': 100.0,
            'flooding': 0.0,
            'total_inflow': 0.5
        }
        mock_engine.get_link_state.return_value = {
            'flow': 0.5,
            'depth': 0.5,
            'volume': 10.0,
            'current_setting': 0.5
        }
        # Mock links dict for direct action application
        mock_link = Mock()
        mock_engine.links = {'P1': mock_link, 'W1': mock_link}
        mock_engine._pending_actions = {}
        mock_engine._step_count = 0

        # Mock sim as iterator for advance()
        mock_sim = MagicMock()
        mock_sim.__iter__ = Mock(return_value=iter([None] * 1000))
        mock_engine.sim = mock_sim

        env = SWMMEnv(config)

        return env

    def test_init(self, env, config):
        """Test environment initialization."""
        assert env.config == config
        assert env.agents == ['pump_1', 'gate_1']
        assert env.possible_agents == ['pump_1', 'gate_1']
        assert env.max_steps == 100

    def test_obs_dims(self, env):
        """Test observation dimensions."""
        assert SWMMEnv.OBS_DIMS['pump'] == 5
        assert SWMMEnv.OBS_DIMS['gate'] == 4
        assert SWMMEnv.OBS_DIMS['weir'] == 4

    def test_get_observation_pump(self, env):
        """Test getting observation for pump agent."""
        obs = env.get_observation('pump_1')

        # Pump observation should have 5 dimensions
        assert obs.shape == (5,)

        # Values should be normalized (near 0 for mean values)
        assert isinstance(obs, np.ndarray)

    def test_get_observation_gate(self, env):
        """Test getting observation for gate agent."""
        obs = env.get_observation('gate_1')

        # Gate observation should have 4 dimensions
        assert obs.shape == (4,)

        assert isinstance(obs, np.ndarray)

    def test_reset(self, env):
        """Test reset method."""
        observations = env.reset()

        # Should return observations for all agents
        assert isinstance(observations, dict)
        assert 'pump_1' in observations
        assert 'gate_1' in observations

        # Engine reset should have been called
        env.engine.reset.assert_called_once()

    def test_step(self, env):
        """Test step method."""
        env.reset()

        actions = {'pump_1': 0.8, 'gate_1': 0.5}

        observations, reward, done, info = env.step(actions)

        # Check return types
        assert isinstance(observations, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        # Check observations
        assert 'pump_1' in observations
        assert 'gate_1' in observations

        # Check action was applied directly to link (last action value)
        # Note: gate_1 maps to W1, and it's applied last
        assert env.engine.links['W1'].target_setting == 0.5

    def test_step_clips_actions(self, env):
        """Test that step clips actions to [0, 1]."""
        env.reset()

        # Test action above 1
        actions = {'pump_1': 1.5}
        env.step(actions)

        # Action should have been clipped to 1.0
        assert env.engine.links['P1'].target_setting == 1.0

        # Test action below 0
        actions = {'pump_1': -0.5}
        env.step(actions)

        # Action should have been clipped to 0.0
        assert env.engine.links['P1'].target_setting == 0.0

    def test_step_termination(self, env):
        """Test step termination conditions."""
        env.reset()

        # Run until max_steps
        for _ in range(env.max_steps + 1):
            actions = {'pump_1': 0.5, 'gate_1': 0.5}
            obs, reward, done, info = env.step(actions)

            if done:
                break

        # Should terminate at max_steps
        assert env._step_count >= env.max_steps

    def test_get_reward(self, env):
        """Test get_reward method."""
        reward = env.get_reward()

        assert isinstance(reward, float)

    def test_get_state(self, env):
        """Test get_state method."""
        state = env.get_state()

        assert 'nodes' in state
        assert 'links' in state
        assert 'rainfall' in state

    def test_close(self, env):
        """Test close method."""
        env.close()

        env.engine.close.assert_called_once()
        assert env.agents == []

    def test_render(self, env, capsys):
        """Test render method."""
        env.reset()
        env.render()

        # Should have printed output
        captured = capsys.readouterr()
        assert 'Step' in captured.out

    def test_check_done_simulation_ended(self, env):
        """Test _check_done when simulation ends."""
        env.engine.is_ended.return_value = True

        done = env._check_done()

        assert done == True

    def test_check_done_max_steps(self, env):
        """Test _check_done when max steps reached."""
        env.engine.is_ended.return_value = False
        env._step_count = env.max_steps

        done = env._check_done()

        assert done == True

    def test_check_done_not_done(self, env):
        """Test _check_done when not done."""
        env.engine.is_ended.return_value = False
        env._step_count = 0

        done = env._check_done()

        assert done == False

    def test_repr(self, env):
        """Test string representation."""
        repr_str = repr(env)

        assert 'SWMMEnv' in repr_str
        assert 'agents=2' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])