"""
Unit tests for SWMMEngine.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swmmEnv.sim.engine import SWMMEngine


class TestSWMMEngine:
    """Test SWMMEngine class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return {
            'inp_file': 'test.inp',
            'time_sync': {
                'decision_interval': 300,
                'swmm_step': 10
            },
            'agents': {
                'pump_1': {
                    'type': 'pump',
                    'link_id': 'P1',
                    'upstream_node': 'J1',
                    'downstream_node': 'J2'
                }
            }
        }

    @pytest.fixture
    def engine(self, mock_config):
        """Create engine instance."""
        return SWMMEngine('test.inp', mock_config)

    def test_init(self, engine, mock_config):
        """Test engine initialization."""
        assert engine.inp_file == 'test.inp'
        assert engine.control_interval == 300
        assert engine._is_started == False
        assert engine._step_count == 0

    def test_apply_action_queues_action(self, engine, mock_config):
        """Test that apply_action queues the action."""
        engine._is_started = True

        # Apply action
        engine.apply_action('pump_1', 0.8)

        # Check action was queued
        assert 'P1' in engine._pending_actions
        assert engine._pending_actions['P1'] == 0.8

    def test_apply_action_invalid_agent(self, engine):
        """Test applying action to invalid agent raises error."""
        with pytest.raises(ValueError, match="Unknown agent"):
            engine.apply_action('invalid_agent', 0.5)

    @patch('swmmEnv.sim.engine.Simulation')
    def test_get_node_state(self, mock_sim_class, engine):
        """Test getting node state."""
        # Mock node
        mock_node = Mock()
        mock_node.depth = 1.5
        mock_node.head = 10.0
        mock_node.volume = 100.0
        mock_node.flooding = 0.0
        mock_node.total_inflow = 0.5

        engine.nodes = {'J1': mock_node}

        state = engine.get_node_state('J1')

        assert state['depth'] == 1.5
        assert state['head'] == 10.0
        assert state['volume'] == 100.0
        assert state['flooding'] == 0.0
        assert state['total_inflow'] == 0.5

    def test_get_node_state_invalid_node(self, engine):
        """Test getting state of invalid node raises error."""
        with pytest.raises(ValueError, match="Unknown node"):
            engine.get_node_state('invalid_node')

    @patch('swmmEnv.sim.engine.Simulation')
    def test_get_link_state(self, mock_sim_class, engine):
        """Test getting link state."""
        mock_link = Mock()
        mock_link.flow = 0.3
        mock_link.depth = 0.5
        mock_link.volume = 10.0
        mock_link.current_setting = 0.8

        engine.links = {'P1': mock_link}

        state = engine.get_link_state('P1')

        assert state['flow'] == 0.3
        assert state['depth'] == 0.5
        assert state['volume'] == 10.0
        assert state['current_setting'] == 0.8

    def test_get_total_flooding(self, engine):
        """Test calculating total flooding."""
        mock_node1 = Mock()
        mock_node1.flooding = 0.1

        mock_node2 = Mock()
        mock_node2.flooding = 0.2

        engine.nodes = {'J1': mock_node1, 'J2': mock_node2}

        total = engine.get_total_flooding()

        assert total == pytest.approx(0.3, abs=1e-6)

    def test_get_rainfall_no_gages(self, engine):
        """Test rainfall when no gages available."""
        engine.raingages = {}

        rainfall = engine.get_rainfall()

        assert rainfall == 0.0

    def test_step_without_start_raises(self, engine):
        """Test stepping without starting raises error."""
        with pytest.raises(RuntimeError, match="not started"):
            engine.step()

    def test_is_ended_not_started(self, engine):
        """Test is_ended returns True when not started."""
        assert engine.is_ended() == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])