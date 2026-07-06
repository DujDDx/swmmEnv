"""
Unit tests for MappingRegistry.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swmmEnv.sim.mapping import MappingRegistry


class TestMappingRegistry:
    """Test MappingRegistry class."""

    @pytest.fixture
    def agents_config(self):
        """Create test agents configuration."""
        return {
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
            },
            'pump_2': {
                'type': 'pump',
                'link_id': 'P2',
                'upstream_node': 'J5',
                'downstream_node': 'J6'
            }
        }

    @pytest.fixture
    def registry(self, agents_config):
        """Create registry instance."""
        return MappingRegistry(agents_config)

    def test_init(self, registry, agents_config):
        """Test initialization."""
        assert registry.agents_config == agents_config
        assert len(registry) == 3

    def test_init_missing_type(self):
        """Test initialization with missing type raises error."""
        config = {
            'agent_1': {
                'link_id': 'P1'
                # Missing 'type'
            }
        }

        with pytest.raises(ValueError, match="missing 'type'"):
            MappingRegistry(config)

    def test_init_missing_link_id(self):
        """Test initialization with missing link_id raises error."""
        config = {
            'agent_1': {
                'type': 'pump'
                # Missing 'link_id'
            }
        }

        with pytest.raises(ValueError, match="missing 'link_id'"):
            MappingRegistry(config)

    def test_init_invalid_type(self):
        """Test initialization with invalid type raises error."""
        config = {
            'agent_1': {
                'type': 'invalid',
                'link_id': 'P1'
            }
        }

        with pytest.raises(ValueError, match="invalid type"):
            MappingRegistry(config)

    def test_get_element_id(self, registry):
        """Test getting element ID."""
        assert registry.get_element_id('pump_1') == 'P1'
        assert registry.get_element_id('gate_1') == 'W1'
        assert registry.get_element_id('pump_2') == 'P2'

    def test_get_element_id_invalid_agent(self, registry):
        """Test getting element ID for invalid agent raises error."""
        with pytest.raises(KeyError, match="Unknown agent"):
            registry.get_element_id('invalid_agent')

    def test_get_element_type(self, registry):
        """Test getting element type."""
        assert registry.get_element_type('pump_1') == 'pump'
        assert registry.get_element_type('gate_1') == 'weir'
        assert registry.get_element_type('pump_2') == 'pump'

    def test_get_upstream_node(self, registry):
        """Test getting upstream node."""
        assert registry.get_upstream_node('pump_1') == 'J1'
        assert registry.get_upstream_node('gate_1') == 'J3'

    def test_get_downstream_node(self, registry):
        """Test getting downstream node."""
        assert registry.get_downstream_node('pump_1') == 'J2'
        assert registry.get_downstream_node('gate_1') == 'J4'

    def test_get_agent_config(self, registry, agents_config):
        """Test getting full agent configuration."""
        config = registry.get_agent_config('pump_1')

        assert config == agents_config['pump_1']

        # Test that returned config is a copy
        config['new_key'] = 'value'
        assert 'new_key' not in registry.agents_config['pump_1']

    def test_get_all_agents(self, registry):
        """Test getting all agent IDs."""
        agents = registry.get_all_agents()

        assert len(agents) == 3
        assert 'pump_1' in agents
        assert 'gate_1' in agents
        assert 'pump_2' in agents

    def test_get_agents_by_type(self, registry):
        """Test getting agents by type."""
        pumps = registry.get_agents_by_type('pump')
        weirs = registry.get_agents_by_type('weir')

        assert len(pumps) == 2
        assert 'pump_1' in pumps
        assert 'pump_2' in pumps

        assert len(weirs) == 1
        assert 'gate_1' in weirs

    def test_get_all_link_ids(self, registry):
        """Test getting all link IDs."""
        link_ids = registry.get_all_link_ids()

        assert len(link_ids) == 3
        assert 'P1' in link_ids
        assert 'W1' in link_ids
        assert 'P2' in link_ids

    def test_get_all_node_ids(self, registry):
        """Test getting all unique node IDs."""
        node_ids = registry.get_all_node_ids()

        assert len(node_ids) == 6
        assert 'J1' in node_ids
        assert 'J2' in node_ids
        assert 'J3' in node_ids
        assert 'J4' in node_ids
        assert 'J5' in node_ids
        assert 'J6' in node_ids

    def test_get_observation_nodes(self, registry):
        """Test getting observation nodes for an agent."""
        nodes = registry.get_observation_nodes('pump_1')

        assert nodes == ['J1', 'J2']

    def test_agent_exists(self, registry):
        """Test checking if agent exists."""
        assert registry.agent_exists('pump_1') == True
        assert registry.agent_exists('invalid') == False

    def test_contains(self, registry):
        """Test __contains__ method."""
        assert 'pump_1' in registry
        assert 'invalid' not in registry

    def test_len(self, registry):
        """Test __len__ method."""
        assert len(registry) == 3

    def test_repr(self, registry):
        """Test string representation."""
        repr_str = repr(registry)

        assert 'MappingRegistry' in repr_str
        assert 'agents=3' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])