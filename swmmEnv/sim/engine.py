"""
SWMMEngine - PySWMM wrapper for reinforcement learning environment.

This module provides a clean interface between PySWMM simulation and RL agents,
handling simulation control, state retrieval, and action application.
"""

import os
from typing import Dict, Optional, Any
import numpy as np

try:
    from pyswmm import Simulation, Nodes, Links, RainGages
    from pyswmm import SystemStats
except ImportError:
    raise ImportError(
        "PySWMM is required for SWMMEngine. "
        "Install with: pip install pyswmm>=2.1.0"
    )


class SWMMEngine:
    """
    Wrapper for PySWMM simulation providing RL-compatible interface.

    This is the sole interaction layer with SWMM simulation. It handles:
    - Simulation initialization and control
    - Action application (pump/gate/weir settings)
    - State observation retrieval
    - Simulation reset via hotstart files

    Example:
        >>> engine = SWMMEngine("model.inp", config)
        >>> engine.start()
        >>> engine.apply_action("pump_1", 0.8)
        >>> engine.step()
        >>> state = engine.get_node_state("J1")
        >>> print(state['depth'])
        >>> engine.close()
    """

    def __init__(self, inp_file: str, config: Dict[str, Any]):
        """
        Initialize SWMMEngine.

        Args:
            inp_file: Path to SWMM .inp file
            config: Configuration dictionary containing:
                - control_interval: Simulation step interval in seconds
                - agents: Agent configuration for mapping
                - hotstart_file: Optional hotstart file path for reset
        """
        self.inp_file = inp_file
        self.config = config

        # Simulation state
        self.sim: Optional[Simulation] = None
        self.nodes: Dict[str, Any] = {}
        self.links: Dict[str, Any] = {}
        self.raingages: Dict[str, Any] = {}

        # Control interval
        self.control_interval = config.get('time_sync', {}).get(
            'decision_interval', 300
        )

        # Hotstart file for reset
        self.hotstart_file = config.get('hotstart_file', None)

        # Simulation tracking
        self._is_started = False
        self._step_count = 0
        self._pending_actions: Dict[str, float] = {}

    def start(self) -> None:
        """
        Start SWMM simulation.

        Creates Simulation object and initializes node/link references.
        Sets step_advance to control_interval for RL control timing.
        """
        if self._is_started:
            self.close()

        # Create simulation
        self.sim = Simulation(self.inp_file)

        # Set control interval for step advancement
        self.sim.step_advance(self.control_interval)

        # Initialize references
        self.nodes = {node.nodeid: node for node in Nodes(self.sim)}
        self.links = {link.linkid: link for link in Links(self.sim)}

        # Initialize raingages if available
        try:
            self.raingages = {gage.raingageid: gage for gage in RainGages(self.sim)}
        except Exception:
            self.raingages = {}

        # Register callbacks for action application and observation retrieval
        self.sim.add_before_step(self._apply_pending_actions)
        self.sim.add_after_step(self._update_step_count)

        # Start simulation iteration (enter context manager)
        self.sim.start()

        # Use hotstart file if provided
        if self.hotstart_file and os.path.exists(self.hotstart_file):
            self.sim.use_hotstart(self.hotstart_file)

        self._is_started = True
        self._step_count = 0

        # Initial step to get first observation
        next(self.sim)

    def step(self) -> None:
        """
        Advance simulation by one control interval.

        This advances by control_interval seconds (set via step_advance).
        """
        if not self._is_started:
            raise RuntimeError("Simulation not started. Call start() first.")

        # Advance simulation
        try:
            next(self.sim)
        except StopIteration:
            # Simulation ended
            pass

    def reset(self) -> None:
        """
        Reset simulation to initial state.

        Uses hotstart file if configured, otherwise restarts simulation.
        """
        self.close()

        if self.hotstart_file and os.path.exists(self.hotstart_file):
            # Restart and load hotstart
            self.start()
            self.sim.use_hotstart(self.hotstart_file)
            self._step_count = 0
            # Initial step after hotstart
            next(self.sim)
        else:
            # Full restart
            self.start()

    def close(self) -> None:
        """
        Close simulation and release resources.
        """
        if self.sim is not None:
            try:
                self.sim.close()
            except Exception:
                pass

        self.sim = None
        self.nodes = {}
        self.links = {}
        self.raingages = {}
        self._is_started = False
        self._step_count = 0
        self._pending_actions = {}

    def apply_action(self, agent_id: str, setting: float) -> None:
        """
        Apply control action to SWMM element.

        Actions are queued and applied in before_step callback.

        Args:
            agent_id: Agent identifier (e.g., "pump_1")
            setting: Control setting value (0.0 to 1.0)
        """
        agent_config = self.config['agents'].get(agent_id)

        if agent_config is None:
            raise ValueError(f"Unknown agent: {agent_id}")

        link_id = agent_config['link_id']

        # Queue action to be applied before next step
        self._pending_actions[link_id] = float(setting)

    def get_node_state(self, node_id: str) -> Dict[str, float]:
        """
        Get state observations for a node.

        Args:
            node_id: Node identifier

        Returns:
            Dictionary with:
                - depth: Water depth above invert (m)
                - head: Water elevation (m)
                - volume: Stored volume (m³)
                - flooding: Flooding rate (m³/s)
                - total_inflow: Total inflow rate (m³/s)
        """
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node: {node_id}")

        node = self.nodes[node_id]

        return {
            'depth': float(node.depth),
            'head': float(node.head),
            'volume': float(node.volume),
            'flooding': float(node.flooding),
            'total_inflow': float(node.total_inflow),
        }

    def get_link_state(self, link_id: str) -> Dict[str, float]:
        """
        Get state observations for a link.

        Args:
            link_id: Link identifier

        Returns:
            Dictionary with:
                - flow: Current flow rate (m³/s)
                - depth: Flow depth (m)
                - volume: Flow volume (m³)
                - current_setting: Current control setting
        """
        if link_id not in self.links:
            raise ValueError(f"Unknown link: {link_id}")

        link = self.links[link_id]

        return {
            'flow': float(link.flow),
            'depth': float(link.depth),
            'volume': float(link.volume),
            'current_setting': float(link.current_setting),
        }

    def get_rainfall(self, gage_id: Optional[str] = None) -> float:
        """
        Get rainfall intensity from raingage.

        Args:
            gage_id: Raingage identifier. If None, returns average.

        Returns:
            Rainfall intensity in mm/h
        """
        if not self.raingages:
            return 0.0

        if gage_id is not None:
            if gage_id in self.raingages:
                return float(self.raingages[gage_id].rainfall)
            return 0.0

        # Average rainfall across all gages
        rainfall_values = [
            float(gage.rainfall) for gage in self.raingages.values()
        ]
        return np.mean(rainfall_values) if rainfall_values else 0.0

    def get_total_flooding(self) -> float:
        """
        Get total flooding rate across all nodes.

        Returns:
            Total flooding rate in m³/s
        """
        total_flooding = sum(
            float(node.flooding) for node in self.nodes.values()
        )
        return total_flooding

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system-wide statistics.

        Returns:
            Dictionary with routing_stats and runoff_stats
        """
        if self.sim is None:
            return {}

        try:
            stats = SystemStats(self.sim)
            return {
                'routing_stats': stats.routing_stats,
                'runoff_stats': stats.runoff_stats,
            }
        except Exception:
            return {}

    def is_ended(self) -> bool:
        """
        Check if simulation has ended.

        Returns:
            True if simulation has reached end time
        """
        if self.sim is None:
            return True

        return not self.sim.sim_is_started

    def get_current_time(self) -> Any:
        """
        Get current simulation time.

        Returns:
            Current simulation datetime
        """
        if self.sim is None:
            return None

        return self.sim.current_time

    def get_step_count(self) -> int:
        """
        Get number of steps executed.

        Returns:
            Step count
        """
        return self._step_count

    def save_hotstart(self, filepath: str) -> None:
        """
        Save current state to hotstart file.

        Args:
            filepath: Path to save hotstart file (.hsf)
        """
        if self.sim is None:
            raise RuntimeError("Simulation not started")

        self.sim.save_hotstart(filepath)
        self.hotstart_file = filepath

    # Internal callback methods

    def _apply_pending_actions(self) -> None:
        """
        Apply queued control actions before simulation step.

        This is registered as a before_step callback.
        """
        for link_id, setting in self._pending_actions.items():
            if link_id in self.links:
                self.links[link_id].target_setting = setting

        # Clear pending actions
        self._pending_actions = {}

    def _update_step_count(self) -> None:
        """
        Update step count after each simulation step.

        This is registered as an after_step callback.
        """
        self._step_count += 1

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False