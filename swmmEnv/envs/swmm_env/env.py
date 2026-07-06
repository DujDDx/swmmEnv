"""
SWMMEnv - Core MDP environment for SWMM reinforcement learning.

This module implements the core reinforcement learning MDP logic,
integrating all simulation components (engine, time_sync, normalizer, mapping).
"""

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np

from swmmEnv.sim.engine import SWMMEngine
from swmmEnv.sim.time_sync import TimeSync
from swmmEnv.sim.normalizer import StateNormalizer
from swmmEnv.sim.mapping import MappingRegistry
from swmmEnv.reward.default_reward import default_reward, get_reward_fn


class SWMMEnv:
    """
    Core MDP environment for SWMM-based reinforcement learning.

    This is the central environment class that integrates:
    - SWMMEngine: PySWMM simulation control
    - TimeSync: RL step synchronization
    - StateNormalizer: Observation/reward normalization
    - MappingRegistry: Agent-to-element mapping
    - Reward function: External reward computation

    This class is independent of MARLlib and can be used standalone
    for testing and custom integration.

    Example:
        >>> config = load_config("config/example.yaml")
        >>> env = SWMMEnv(config)
        >>> obs = env.reset()
        >>> actions = {"pump_1": 0.8, "gate_1": 0.5}
        >>> obs, reward, done, info = env.step(actions)
        >>> env.close()
    """

    # Observation dimensions by agent type
    OBS_DIMS = {
        'pump': 5,      # upstream_depth, downstream_depth, flow, setting, rainfall
        'gate': 4,      # upstream_depth, downstream_depth, setting, rainfall
        'weir': 4,      # same as gate
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SWMMEnv.

        Args:
            config: Configuration dictionary containing:
                - inp_file: Path to SWMM .inp file
                - agents: Agent configuration
                - time_sync: Time synchronization parameters
                - normalization: Normalization parameters
                - reward_fn: Reward function name or callable
                - max_steps: Maximum episode steps
                - obs_nodes: Nodes to observe
                - obs_raingage: Raingage ID for rainfall observation
        """
        self.config = config

        # Initialize components
        self.engine = SWMMEngine(config['inp_file'], config)

        time_sync_config = config['time_sync']
        self.time_sync = TimeSync(
            decision_interval=time_sync_config['decision_interval'],
            swmm_step=time_sync_config['swmm_step']
        )

        self.normalizer = StateNormalizer(config['normalization'])

        self.mapping = MappingRegistry(config['agents'])

        # Load reward function
        reward_fn = config.get('reward_fn', 'default_reward')

        if isinstance(reward_fn, str):
            self.reward_fn = get_reward_fn(reward_fn)
        elif callable(reward_fn):
            self.reward_fn = reward_fn
        else:
            self.reward_fn = default_reward

        # Episode tracking
        self.max_steps = config.get('max_steps', 1000)
        self.warmup_steps = config.get('warmup_steps', 0)
        self._step_count = 0

        # Agents list
        self.agents = self.mapping.get_all_agents()
        self.possible_agents = self.agents.copy()

        # Observation configuration
        self.obs_nodes = config.get('obs_nodes', [])
        self.obs_raingage = config.get('obs_raingage', None)

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment for new episode.

        Returns:
            Dictionary of initial observations for each agent
        """
        # Reset simulation engine
        self.engine.reset()

        # Reset time sync
        self.time_sync.reset()

        # Reset agents
        self.agents = self.possible_agents.copy()

        # Reset tracking
        self._step_count = 0

        # Warmup steps (if configured)
        if self.warmup_steps > 0:
            for _ in range(self.warmup_steps):
                self.engine.step()

        # Get initial observations
        observations = {
            agent: self.get_observation(agent)
            for agent in self.agents
        }

        return observations

    def step(
        self,
        action_dict: Dict[str, float]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute one RL step.

        Args:
            action_dict: Dictionary mapping agent_id to action value
                        (0.0 to 1.0 for pump/gate/weir settings)

        Returns:
            Tuple of:
                - observations: Dict of agent observations
                - reward: Global reward value
                - done: Episode termination flag
                - info: Additional information dict
        """
        # Apply actions to simulation
        for agent_id, action in action_dict.items():
            if agent_id in self.agents:
                # Clip action to valid range
                action = np.clip(action, 0.0, 1.0)
                self.engine.apply_action(agent_id, float(action))

        # Advance simulation by decision interval
        self.time_sync.advance(self.engine)

        # Get observations
        observations = {
            agent: self.get_observation(agent)
            for agent in self.agents
        }

        # Compute global reward
        reward = self.get_reward()
        reward = self.normalizer.normalize_reward(reward)

        # Check termination
        done = self._check_done()

        # Info dict
        info = {
            'step_count': self._step_count,
            'elapsed_time': self.time_sync.get_elapsed_time(),
            'total_flooding': self.engine.get_total_flooding(),
            'simulation_time': str(self.engine.get_current_time()),
        }

        self._step_count += 1

        if done:
            self.agents = []

        return observations, reward, done, info

    def get_observation(self, agent_id: str) -> np.ndarray:
        """
        Get observation for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Normalized observation array
        """
        agent_config = self.mapping.get_agent_config(agent_id)
        agent_type = agent_config['type']

        # Get rainfall observation
        rainfall = self.engine.get_rainfall(self.obs_raingage)

        if agent_type == 'pump':
            # Pump observation: upstream_depth, downstream_depth, flow, setting, rainfall
            upstream_node = agent_config.get('upstream_node')
            downstream_node = agent_config.get('downstream_node')
            link_id = agent_config['link_id']

            upstream_depth = (
                self.engine.get_node_state(upstream_node)['depth']
                if upstream_node else 0.0
            )
            downstream_depth = (
                self.engine.get_node_state(downstream_node)['depth']
                if downstream_node else 0.0
            )
            link_state = self.engine.get_link_state(link_id)
            flow = link_state['flow']
            setting = link_state['current_setting']

            raw_obs = np.array([
                upstream_depth,
                downstream_depth,
                flow,
                setting,
                rainfall
            ], dtype=np.float32)

            # Normalize
            obs_names = ['depth', 'depth', 'flow', 'setting', 'rainfall']
            normalized_obs = self.normalizer.normalize_obs(raw_obs, obs_names)

        else:  # gate or weir
            # Observation: upstream_depth, downstream_depth, setting, rainfall
            upstream_node = agent_config.get('upstream_node')
            downstream_node = agent_config.get('downstream_node')
            link_id = agent_config['link_id']

            upstream_depth = (
                self.engine.get_node_state(upstream_node)['depth']
                if upstream_node else 0.0
            )
            downstream_depth = (
                self.engine.get_node_state(downstream_node)['depth']
                if downstream_node else 0.0
            )
            link_state = self.engine.get_link_state(link_id)
            setting = link_state['current_setting']

            raw_obs = np.array([
                upstream_depth,
                downstream_depth,
                setting,
                rainfall
            ], dtype=np.float32)

            # Normalize
            obs_names = ['depth', 'depth', 'setting', 'rainfall']
            normalized_obs = self.normalizer.normalize_obs(raw_obs, obs_names)

        return normalized_obs

    def get_reward(self) -> float:
        """
        Compute global reward.

        Returns:
            Unnormalized reward value
        """
        return self.reward_fn(self.engine, self.config)

    def _check_done(self) -> bool:
        """
        Check if episode should terminate.

        Returns:
            True if episode should end
        """
        # Simulation ended
        if self.engine.is_ended():
            return True

        # Max steps reached
        if self._step_count >= self.max_steps:
            return True

        return False

    def get_state(self) -> Dict[str, Any]:
        """
        Get full state of all observed nodes and links.

        Returns:
            Dictionary with node_states and link_states
        """
        node_states = {}
        for node_id in self.obs_nodes:
            try:
                node_states[node_id] = self.engine.get_node_state(node_id)
            except (KeyError, ValueError):
                node_states[node_id] = None

        link_states = {}
        for link_id in self.mapping.get_all_link_ids():
            try:
                link_states[link_id] = self.engine.get_link_state(link_id)
            except (KeyError, ValueError):
                link_states[link_id] = None

        return {
            'nodes': node_states,
            'links': link_states,
            'rainfall': self.engine.get_rainfall(self.obs_raingage),
        }

    def close(self) -> None:
        """
        Close environment and release resources.
        """
        self.engine.close()
        self.agents = []

    def render(self, mode: str = 'human') -> None:
        """
        Render environment state.

        Args:
            mode: Render mode (currently only 'human' text output)
        """
        if mode == 'human':
            print(f"\n=== Step {self._step_count} ===")
            print(f"Time: {self.engine.get_current_time()}")
            print(f"Elapsed: {self.time_sync.get_elapsed_time_minutes():.1f} min")
            print(f"Total flooding: {self.engine.get_total_flooding():.3f} m³/s")
            print(f"Rainfall: {self.engine.get_rainfall(self.obs_raingage):.1f} mm/h")

            for agent in self.possible_agents:
                obs = self.get_observation(agent)
                print(f"  {agent}: obs = {obs}")

            for node_id in self.obs_nodes[:4]:  # Limit output
                try:
                    state = self.engine.get_node_state(node_id)
                    print(f"  Node {node_id}: depth={state['depth']:.2f}m, "
                          f"flooding={state['flooding']:.3f}m³/s")
                except (KeyError, ValueError):
                    pass

    def __repr__(self) -> str:
        return (
            f"SWMMEnv(agents={len(self.possible_agents)}, "
            f"max_steps={self.max_steps}, "
            f"inp_file='{self.config.get('inp_file', 'N/A')}')"
        )