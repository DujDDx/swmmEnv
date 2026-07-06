"""
MappingRegistry - Registry for agent to SWMM element mapping.

This module manages the mapping between RL agent identifiers and
SWMM model elements (pumps, gates, weirs, nodes).
"""

from typing import Dict, Any, List, Optional


class MappingRegistry:
    """
    Registry for mapping between agent IDs and SWMM elements.

    Provides structured access to agent configuration, including
    element type, link/node IDs, and connectivity information.

    Example:
        >>> registry = MappingRegistry(agents_config)
        >>> element_id = registry.get_element_id("pump_1")
        >>> element_type = registry.get_element_type("pump_1")
        >>> upstream_node = registry.get_upstream_node("pump_1")
    """

    # Supported agent types
    VALID_TYPES = {'pump', 'gate', 'weir'}

    def __init__(self, agents_config: Dict[str, Dict[str, Any]]):
        """
        Initialize MappingRegistry.

        Args:
            agents_config: Dictionary mapping agent_id to configuration:
                {
                    "pump_1": {
                        "type": "pump",
                        "link_id": "P1",
                        "upstream_node": "J1",
                        "downstream_node": "J2"
                    },
                    "gate_1": {
                        "type": "weir",
                        "link_id": "W1",
                        "upstream_node": "J3",
                        "downstream_node": "J4"
                    }
                }

        Raises:
            ValueError: If configuration is invalid
        """
        self.agents_config = agents_config
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate agent configuration.

        Raises:
            ValueError: If required fields are missing or invalid
        """
        for agent_id, config in self.agents_config.items():
            # Check required fields
            if 'type' not in config:
                raise ValueError(f"Agent {agent_id}: missing 'type' field")

            if 'link_id' not in config:
                raise ValueError(f"Agent {agent_id}: missing 'link_id' field")

            # Check type is valid
            agent_type = config['type']
            if agent_type not in self.VALID_TYPES:
                raise ValueError(
                    f"Agent {agent_id}: invalid type '{agent_type}'. "
                    f"Must be one of {self.VALID_TYPES}"
                )

    def get_element_id(self, agent_id: str) -> str:
        """
        Get SWMM element ID for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            SWMM link ID

        Raises:
            KeyError: If agent_id not found
        """
        if agent_id not in self.agents_config:
            raise KeyError(f"Unknown agent: {agent_id}")

        return self.agents_config[agent_id]['link_id']

    def get_element_type(self, agent_id: str) -> str:
        """
        Get element type for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Element type ('pump', 'gate', or 'weir')

        Raises:
            KeyError: If agent_id not found
        """
        if agent_id not in self.agents_config:
            raise KeyError(f"Unknown agent: {agent_id}")

        return self.agents_config[agent_id]['type']

    def get_upstream_node(self, agent_id: str) -> Optional[str]:
        """
        Get upstream node ID for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Upstream node ID or None if not configured
        """
        if agent_id not in self.agents_config:
            raise KeyError(f"Unknown agent: {agent_id}")

        return self.agents_config[agent_id].get('upstream_node')

    def get_downstream_node(self, agent_id: str) -> Optional[str]:
        """
        Get downstream node ID for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Downstream node ID or None if not configured
        """
        if agent_id not in self.agents_config:
            raise KeyError(f"Unknown agent: {agent_id}")

        return self.agents_config[agent_id].get('downstream_node')

    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """
        Get full configuration for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent configuration dictionary

        Raises:
            KeyError: If agent_id not found
        """
        if agent_id not in self.agents_config:
            raise KeyError(f"Unknown agent: {agent_id}")

        return self.agents_config[agent_id].copy()

    def get_all_agents(self) -> List[str]:
        """
        Get list of all agent IDs.

        Returns:
            List of agent identifiers
        """
        return list(self.agents_config.keys())

    def get_agents_by_type(self, agent_type: str) -> List[str]:
        """
        Get agents of a specific type.

        Args:
            agent_type: Element type ('pump', 'gate', 'weir')

        Returns:
            List of agent IDs of the specified type
        """
        return [
            agent_id for agent_id, config in self.agents_config.items()
            if config.get('type') == agent_type
        ]

    def get_all_link_ids(self) -> List[str]:
        """
        Get all SWMM link IDs.

        Returns:
            List of link IDs
        """
        return [
            config['link_id'] for config in self.agents_config.values()
        ]

    def get_all_node_ids(self) -> List[str]:
        """
        Get all unique node IDs referenced by agents.

        Returns:
            List of unique node IDs
        """
        nodes = set()
        for config in self.agents_config.values():
            if 'upstream_node' in config:
                nodes.add(config['upstream_node'])
            if 'downstream_node' in config:
                nodes.add(config['downstream_node'])

        return list(nodes)

    def get_observation_nodes(self, agent_id: str) -> List[str]:
        """
        Get nodes that should be observed for an agent.

        Typically includes upstream and downstream nodes.

        Args:
            agent_id: Agent identifier

        Returns:
            List of node IDs to observe
        """
        if agent_id not in self.agents_config:
            raise KeyError(f"Unknown agent: {agent_id}")

        config = self.agents_config[agent_id]
        nodes = []

        if 'upstream_node' in config:
            nodes.append(config['upstream_node'])

        if 'downstream_node' in config:
            nodes.append(config['downstream_node'])

        return nodes

    def agent_exists(self, agent_id: str) -> bool:
        """
        Check if agent exists in registry.

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent exists
        """
        return agent_id in self.agents_config

    def __len__(self) -> int:
        """Get number of registered agents."""
        return len(self.agents_config)

    def __contains__(self, agent_id: str) -> bool:
        """Check if agent is in registry."""
        return agent_id in self.agents_config

    def __repr__(self) -> str:
        agent_types = {}
        for config in self.agents_config.values():
            t = config.get('type', 'unknown')
            agent_types[t] = agent_types.get(t, 0) + 1

        return (
            f"MappingRegistry(agents={len(self)}, "
            f"types={agent_types})"
        )