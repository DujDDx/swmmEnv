"""
SWMMEngine - PySWMM wrapper for reinforcement learning environment.

This module provides a clean interface between PySWMM simulation and RL agents,
handling simulation control, state retrieval, and action application.

IMPORTANT: PySWMM has limitations on multiple concurrent simulations.
For RLlib parallel training, use worker_index to create separate instances
or use hotstart files for efficient resets.
"""

import os
import shutil
import tempfile
from typing import Dict, Optional, Any
import numpy as np
import threading

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

    KEY FEATURES:
    - Thread-safe operation with worker_index for parallel environments
    - Hotstart file support for efficient episode resets
    - Automatic inp file copying for worker isolation

    Example:
        >>> engine = SWMMEngine("model.inp", config, worker_index=0)
        >>> engine.start()
        >>> engine.apply_action("pump_1", 0.8)
        >>> engine.step()
        >>> state = engine.get_node_state("J1")
        >>> print(state['depth'])
        >>> engine.close()
    """

    # Global lock for PySWMM operations (safety measure)
    _global_lock = threading.Lock()

    # Track active simulations to prevent conflicts
    _active_simulations = {}

    def __init__(
        self,
        inp_file: str,
        config: Dict[str, Any],
        worker_index: int = 0,
        copy_inp: bool = True
    ):
        """
        Initialize SWMMEngine.

        Args:
            inp_file: Path to SWMM .inp file
            config: Configuration dictionary containing:
                - control_interval: Simulation step interval in seconds
                - agents: Agent configuration for mapping
                - hotstart_file: Optional hotstart file path for reset
            worker_index: Worker index for parallel environments (default 0)
            copy_inp: Whether to copy inp file for worker isolation (default True)
        """
        self.original_inp_file = inp_file
        self.config = config
        self.worker_index = worker_index

        # Copy inp file for this worker to avoid conflicts
        if copy_inp and worker_index > 0:
            # Create worker-specific temp directory
            worker_dir = tempfile.mkdtemp(prefix=f"swmm_worker_{worker_index}_")
            worker_inp = os.path.join(worker_dir, os.path.basename(inp_file))
            shutil.copy2(inp_file, worker_inp)
            self.inp_file = worker_inp
            self._worker_dir = worker_dir
        else:
            self.inp_file = inp_file
            self._worker_dir = None

        # Simulation state
        self.sim: Optional[Simulation] = None
        self.nodes: Dict[str, Any] = {}
        self.links: Dict[str, Any] = {}
        self.raingages: Dict[str, Any] = {}

        # Control interval
        self.control_interval = config.get('time_sync', {}).get(
            'decision_interval', 300
        )

        # Hotstart file management
        self.hotstart_file = config.get('hotstart_file', None)
        self._initial_hotstart = None  # Saved initial state
        self._hotstart_created = False

        # Simulation tracking
        self._is_started = False
        self._step_count = 0
        self._pending_actions: Dict[str, float] = {}

    def start(self) -> None:
        """
        Start SWMM simulation.

        Creates Simulation object and initializes node/link references.
        Sets step_advance to control_interval for RL control timing.

        THREAD SAFETY:
        - Uses global lock for PySWMM operations
        - Registers simulation in _active_simulations dict
        """
        if self._is_started:
            self.close()

        # Use global lock for thread safety
        with self._global_lock:
            # Check if this worker already has active simulation
            if self.worker_index in self._active_simulations:
                # Close previous simulation first
                try:
                    self._active_simulations[self.worker_index].close()
                except Exception:
                    pass

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

            # Register in active simulations
            self._active_simulations[self.worker_index] = self.sim

            # Create initial hotstart if not provided
            if self.hotstart_file is None and not self._hotstart_created:
                # Save initial state for efficient reset
                self._create_initial_hotstart()

            # Use hotstart file if provided
            if self.hotstart_file and os.path.exists(self.hotstart_file):
                self.sim.use_hotstart(self.hotstart_file)

            self._is_started = True
            self._step_count = 0

            # Initial step to get first observation
            next(self.sim)

    def _create_initial_hotstart(self) -> None:
        """
        Create initial hotstart file for efficient resets.

        This saves the simulation state at step 0, allowing fast
        episode resets without recreating the simulation.
        """
        if self._worker_dir:
            hotstart_path = os.path.join(
                self._worker_dir,
                f"initial_state_{self.worker_index}.hsf"
            )
        else:
            hotstart_path = os.path.join(
                tempfile.gettempdir(),
                f"swmm_initial_{self.worker_index}.hsf"
            )

        try:
            # Run initial warmup if needed
            warmup = self.config.get('warmup_steps', 0)
            if warmup > 0:
                for _ in range(warmup):
                    next(self.sim)

            # Save initial state
            self.sim.save_hotstart(hotstart_path)
            self._initial_hotstart = hotstart_path
            self._hotstart_created = True
        except Exception as e:
            print(f"Warning: Could not create hotstart file: {e}")
            self._initial_hotstart = None

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
        Reset simulation to initial state efficiently.

        Uses hotstart file for fast reset without recreating simulation.
        This is much faster than closing and restarting simulation.

        EFFICIENCY:
        - Uses saved hotstart file instead of recreating simulation
        - Only recreates simulation if hotstart is unavailable
        """
        # Prefer using hotstart for efficiency
        if self._initial_hotstart and os.path.exists(self._initial_hotstart):
            # Efficient reset using hotstart
            self._reset_with_hotstart()
        elif self.hotstart_file and os.path.exists(self.hotstart_file):
            # Use provided hotstart file
            self._reset_with_hotstart()
        else:
            # Fallback: recreate simulation
            self.close()
            self.start()

    def _reset_with_hotstart(self) -> None:
        """
        Reset simulation using hotstart file (fast).

        This avoids recreating the simulation object, significantly
        improving reset efficiency for RL training.
        """
        if not self._is_started:
            self.start()
            return

        hotstart_file = self._initial_hotstart or self.hotstart_file

        with self._global_lock:
            try:
                # Load hotstart state
                self.sim.use_hotstart(hotstart_file)
                self._step_count = 0
                self._pending_actions = {}

                # Reset agent references (nodes/links still valid)
                # They persist across hotstart loads

            except Exception as e:
                # Fallback to recreate if hotstart fails
                print(f"Warning: Hotstart reset failed: {e}, recreating simulation")
                self.close()
                self.start()

    def close(self) -> None:
        """
        Close simulation and release resources.

        Cleans up worker-specific files and removes from active simulations.
        """
        with self._global_lock:
            if self.sim is not None:
                try:
                    self.sim.close()
                except Exception:
                    pass

            # Remove from active simulations
            if self.worker_index in self._active_simulations:
                try:
                    del self._active_simulations[self.worker_index]
                except Exception:
                    pass

        self.sim = None
        self.nodes = {}
        self.links = {}
        self.raingages = {}
        self._is_started = False
        self._step_count = 0
        self._pending_actions = {}

        # Clean up worker directory if created
        if self._worker_dir and os.path.exists(self._worker_dir):
            try:
                shutil.rmtree(self._worker_dir)
            except Exception:
                pass

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