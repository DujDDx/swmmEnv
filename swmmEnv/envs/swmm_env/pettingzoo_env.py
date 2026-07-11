"""
SWMMParallelEnv - PettingZoo ParallelEnv wrapper for SWMMEnv.

This module provides a PettingZoo-compatible multi-agent environment
that wraps SWMMEnv, making it directly usable with MARLlib.

RLlib COMPATIBILITY:
- Implements ParallelEnv API correctly
- Provides proper observation_space and action_space properties
- Returns correct types from step() for RLlib wrappers
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    from pettingzoo import ParallelEnv
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "PettingZoo and gymnasium are required for SWMMParallelEnv. "
        "Install with: pip install pettingzoo gymnasium"
    )

from swmmEnv.envs.swmm_env.env import SWMMEnv


class SWMMParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for SWMM-based multi-agent RL.

    This is the interface layer for MARLlib training. It provides:
    - Standard PettingZoo ParallelEnv API
    - Observation and action space definitions per agent
    - Global reward shared across all agents
    - Agent management (active/terminated agents)

    RLLIB COMPATIBILITY:
    - observation_space property returns the space for any agent
    - action_space property returns the space for any agent
    - step() returns correct types for RLlib wrappers

    Example:
        >>> from swmmEnv import SWMMParallelEnv, load_config
        >>> config = load_config("config/example.yaml")
        >>> env = SWMMParallelEnv(config)
        >>> observations, info = env.reset()
        >>> actions = {"pump_1": np.array([0.8]), "gate_1": np.array([0.5])}
        >>> obs, rewards, terms, truncs, infos = env.step(actions)
        >>> env.close()

    For MARLlib integration:
        >>> from marllib import marl
        >>> from swmmEnv.envs.register_env import make_env
        >>> env = make_env(environment_name="swmm", map_name="control")
        >>> # Then use with MAPPO or other MARL algorithms
    """

    metadata = {
        "render_modes": ["human"],
        "name": "swmm_env_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SWMMParallelEnv.

        Args:
            config: Configuration dictionary for SWMMEnv
        """
        # Create core environment
        self.core_env = SWMMEnv(config)

        # Copy config for reference
        self.config = config

        # Parse action space configuration
        action_cfg = config.get('action_space', {})
        self.action_space_type = action_cfg.get('type', 'continuous')
        self.action_space_params = action_cfg  # keep full dict for _setup_spaces

        # Agents (from core env)
        self.agents = self.core_env.agents.copy()
        self.possible_agents = self.core_env.possible_agents.copy()

        # Define observation and action spaces per agent
        self._setup_spaces()

        # Episode tracking
        self._has_reset = False

    def _setup_spaces(self) -> None:
        """
        Define observation and action spaces for each agent based on config.

        Action space is driven by config['action_space']:
        - type == 'discrete' -> spaces.Discrete(n)
        - type == 'continuous' (default) -> spaces.Box(low, high, shape)

        Observation dimensions are read from core_env._obs_dims (computed from
        observation config) or fall back to class-level SWMMEnv.OBS_DIMS.
        """
        self.observation_spaces = {}
        self.action_spaces = {}

        for agent_id in self.possible_agents:
            agent_type = self.core_env.mapping.get_element_type(agent_id)

            # Use dynamic obs_dims if available, otherwise fall back to class-level static
            if hasattr(self.core_env, '_obs_dims'):
                obs_dim = self.core_env._obs_dims.get(agent_type, SWMMEnv.OBS_DIMS.get(agent_type, 4))
            else:
                obs_dim = SWMMEnv.OBS_DIMS.get(agent_type, 4)

            # Observation space: continuous values (normalized)
            # Using reasonable bounds after normalization
            self.observation_spaces[agent_id] = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(obs_dim,),
                dtype=np.float32
            )

            # Action space: configurable via config['action_space']
            cfg = self.action_space_params
            if cfg.get('type', 'continuous') == 'discrete':
                n = cfg.get('n', 11)
                self.action_spaces[agent_id] = spaces.Discrete(n)
            else:
                low = cfg.get('low', 0.0)
                high = cfg.get('high', 1.0)
                shape = tuple(cfg.get('shape', [1]))
                self.action_spaces[agent_id] = spaces.Box(
                    low=low, high=high, shape=shape, dtype=np.float32
                )

    def observation_space(self, agent: str) -> spaces.Box:
        """
        Get observation space for a specific agent (RLlib-compatible).

        Args:
            agent: Agent identifier

        Returns:
            Observation space (Box) for the specified agent
        """
        if agent not in self.observation_spaces:
            raise ValueError(f"Unknown agent: {agent}")
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Union[spaces.Box, spaces.Discrete]:
        """
        Get action space for a specific agent (RLlib-compatible).

        Args:
            agent: Agent identifier

        Returns:
            Action space (Box for continuous, Discrete for discrete)
            for the specified agent
        """
        if agent not in self.action_spaces:
            raise ValueError(f"Unknown agent: {agent}")
        return self.action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed (for reproducibility)
            options: Additional reset options

        Returns:
            Tuple of:
                - observations: Dict mapping agent_id to observation
                - infos: Dict mapping agent_id to info dict
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset core environment
        observations = self.core_env.reset()

        # Reset agent list
        self.agents = self.possible_agents.copy()

        # Create info dict
        infos = {agent: {} for agent in self.agents}

        self._has_reset = True

        return observations, infos

    def step(
        self,
        actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]]
    ]:
        """
        Execute one step with actions from all agents.

        Args:
            actions: Dict mapping agent_id to action array
                    Each action is shape (1,) with value in [0, 1]

        Returns:
            Tuple of:
                - observations: Dict of agent observations (np.ndarray)
                - rewards: Dict of agent rewards (float, global reward shared)
                - terminations: Dict of agent termination flags (bool)
                - truncations: Dict of agent truncation flags (bool)
                - infos: Dict of agent info dicts

        RLLIB COMPATIBILITY:
        - Returns correct types: observations are np.ndarray, not dict
        - Rewards are floats, not arrays
        - Terminations and truncations are booleans
        """
        if not self._has_reset:
            raise RuntimeError(
                "Environment must be reset before stepping. Call reset() first."
            )

        # Convert actions to scalar values
        action_dict = {}
        for agent_id, action_array in actions.items():
            if agent_id in self.agents:
                # Handle both array and scalar actions
                if isinstance(action_array, np.ndarray):
                    action_value = float(action_array.flatten()[0])
                else:
                    action_value = float(action_array)
                action_dict[agent_id] = action_value

        # Step core environment
        observations, global_reward, done, info = self.core_env.step(action_dict)

        # Ensure observations are numpy arrays
        for agent_id in observations:
            if not isinstance(observations[agent_id], np.ndarray):
                observations[agent_id] = np.array(observations[agent_id], dtype=np.float32)

        # Create return dictionaries
        # All agents share global reward (SWMM is coupled system)
        rewards = {agent: float(global_reward) for agent in self.agents}

        # Termination: all agents terminate together
        terminations = {agent: bool(done) for agent in self.agents}

        # Truncation: not used, always False
        truncations = {agent: False for agent in self.agents}

        # Info: same info for all agents
        infos = {agent: info.copy() for agent in self.agents}

        # Update agent list if terminated
        if done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def observe(self, agent: str) -> np.ndarray:
        """
        Get observation for a specific agent.

        Args:
            agent: Agent identifier

        Returns:
            Observation array for the agent
        """
        if agent not in self.possible_agents:
            raise ValueError(f"Unknown agent: {agent}")

        if agent not in self.agents:
            # Agent terminated, return zeros
            obs_dim = self.observation_spaces[agent].shape[0]
            return np.zeros(obs_dim, dtype=np.float32)

        return self.core_env.get_observation(agent)

    def state(self) -> np.ndarray:
        """
        Get global state (for centralized critic algorithms).

        Returns:
            Flattened observation array across all agents
        """
        # Concatenate all agent observations
        all_obs = [
            self.observe(agent) for agent in self.possible_agents
        ]

        return np.concatenate(all_obs)

    def render(self, mode: str = 'human') -> None:
        """
        Render environment state.

        Args:
            mode: Render mode ('human' for text output)
        """
        self.core_env.render(mode)

    def close(self) -> None:
        """
        Close environment and release resources.
        """
        self.core_env.close()
        self.agents = []

    def get_env_info(self) -> Dict[str, Any]:
        """
        Get environment information for MARLlib.

        Returns:
            Dictionary with:
                - space_obs: Observation space dict
                - space_act: Action space dict
                - num_agents: Number of agents
                - episode_limit: Maximum episode steps
                - policy_mapping_info: Policy mapping configuration
        """
        return {
            'space_obs': self.observation_spaces,
            'space_act': self.action_spaces,
            'num_agents': len(self.possible_agents),
            'episode_limit': self.config.get('max_steps', 1000),
            'policy_mapping_info': {
                'all_scenario': {
                    'description': 'SWMM stormwater control',
                    'team_prefix': ('pump_', 'gate_', 'weir_'),
                    'all_agents_one_policy': True,
                    'one_agent_one_policy': True,
                }
            }
        }

    @property
    def num_agents(self) -> int:
        """
        Get number of active agents.

        Returns:
            Number of currently active agents
        """
        return len(self.agents)

    def __repr__(self) -> str:
        return (
            f"SWMMParallelEnv(agents={len(self.possible_agents)}, "
            f"inp_file='{self.config.get('inp_file', 'N/A')}')"
        )