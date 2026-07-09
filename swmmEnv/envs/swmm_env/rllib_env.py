"""
SWMMMultiAgentEnv - RLlib MultiAgentEnv adapter for SWMMEnv.

This module provides an RLlib-compatible multi-agent environment that wraps
SWMMParallelEnv (PettingZoo ParallelEnv), so users can pass the class directly
to RLlib without writing a manual adapter.

Usage with RLlib (newer API — pass class reference):
    >>> from swmmEnv import SWMMMultiAgentEnv
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>>
    >>> config = PPOConfig().environment(
    ...     SWMMMultiAgentEnv,
    ...     env_config={"config_path": "configs/control.yaml"},
    ... )
    >>> algo = config.build()

Direct usage (for debugging / testing):
    >>> from swmmEnv import SWMMMultiAgentEnv, load_config
    >>> env = SWMMMultiAgentEnv(load_config("configs/control.yaml"))
    >>> obs, info = env.reset()
    >>> actions = {aid: np.array([0.5]) for aid in env.possible_agents}
    >>> obs, rewards, terms, truncs, infos = env.step(actions)
    >>> env.close()
"""

from typing import Dict, Any, Optional, Tuple, Set, List
import numpy as np
from gymnasium import spaces

# Lazy import — ray[rllib] is an optional (marl) dependency.
# If ray is not installed, the class falls back to inheriting from object.
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as _MultiAgentEnv
except ImportError:
    _MultiAgentEnv = object

from swmmEnv.config.loader import load_config


class SWMMMultiAgentEnv(_MultiAgentEnv):
    """
    RLlib-compatible MultiAgentEnv wrapping SWMMParallelEnv.

    Translates between the PettingZoo ParallelEnv API (used by SWMMParallelEnv)
    and RLlib's MultiAgentEnv API.  The two are nearly identical, but RLlib
    requires:

    - ``observation_spaces`` / ``action_spaces`` as plain ``dict`` of
      agent ID → ``gymnasium.Space``  (new API stack)
    - ``observation_space`` / ``action_space`` as ``gymnasium.spaces.Dict``
      (legacy OldAPIStack — also set for backward compatibility)
    - ``possible_agents`` property (replaces deprecated ``get_agent_ids()``)
    - ``reset()`` returning ``(obs_dict, info_dict)`` tuple
    - ``step()`` returning a 5-tuple with an ``'__all__'`` key in the
      terminated / truncated dicts

    This class can be passed directly to RLlib's algorithm config builder::

        PPOConfig().environment(SWMMMultiAgentEnv, env_config={...})
    """

    def __init__(self, env_config: Dict[str, Any]):

        # --- Handle RLlib-style env_config wrapper ---------------------------
        # RLlib may pass {"config_path": "..."} when configured via the
        # algorithm builder.  Resolve it to a full config dict here so the
        # caller doesn't have to.
        if isinstance(env_config, dict) and "config_path" in env_config:
            env_config = load_config(env_config["config_path"],merge_defaults=False)

        self._env_config = env_config

        # Import here to keep ray[rllib] an optional dependency at package
        # level (the user only needs it when they create this class).
        from swmmEnv.envs.swmm_env.pettingzoo_env import SWMMParallelEnv

        self._pz_env = SWMMParallelEnv(env_config)
        self._agent_ids: List[str] = list(self._pz_env.possible_agents)
        super().__init__()
        # --- Build per-agent spaces (new RLlib API stack) --------------------
        self.observation_spaces: Dict[str, spaces.Space] = {
            aid: self._pz_env.observation_spaces[aid] for aid in self._agent_ids
        }
        self.action_spaces: Dict[str, spaces.Space] = {
            aid: self._pz_env.action_spaces[aid] for aid in self._agent_ids
        }

        # --- Build Dict spaces (legacy OldAPIStack) -------------------------
        self.observation_space: spaces.Dict = spaces.Dict(
            self.observation_spaces
        )
        self.action_space: spaces.Dict = spaces.Dict(
            self.action_spaces
        )

        # Skip RLlib's environment checking — the PettingZoo env underneath
        # already validates its own interface.
        self._skip_env_checking = True

    # ------------------------------------------------------------------
    # RLlib MultiAgentEnv interface
    # ------------------------------------------------------------------

    @property
    def possible_agents(self) -> List[str]:
        """Return all agent IDs that may appear in the environment (RLlib convention)."""
        return self._agent_ids

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Reset the environment.

        Returns:
            (obs_dict, info_dict) — each keyed by agent ID.
        """
        return self._pz_env.reset(seed=seed, options=options)

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Apply actions and advance the simulation by one decision interval.

        Args:
            action_dict: Mapping ``{agent_id: np.ndarray}`` with shape ``(1,)``
                         per agent.

        Returns:
            ``(obs, rewards, terminated, truncated, infos)`` — standard
            RLlib MultiAgentEnv 5-tuple.  The ``terminated`` and ``truncated``
            dicts include the required ``'__all__'`` key indicating whether
            the entire episode is done (all agents terminated/truncated).
        """
        obs, rewards, terms, truncs, infos = self._pz_env.step(action_dict)

        # RLlib MultiAgentEnv requires '__all__' in both terminated and
        # truncated dicts.  Follow the same convention as RLlib's own
        # ParallelPettingZooEnv wrapper: AND all agent-level values.
        terms["__all__"] = bool(all(terms.values()))
        truncs["__all__"] = bool(all(truncs.values()))

        return obs, rewards, terms, truncs, infos

    # ------------------------------------------------------------------
    # Additional utility methods
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release SWMM simulation resources."""
        if hasattr(self, "_pz_env"):
            self._pz_env.close()

    def render(self, mode: str = "human") -> None:
        """Print a text summary of the current simulation state."""
        self._pz_env.render(mode)

    def __repr__(self) -> str:
        return (
            f"SWMMMultiAgentEnv(agents={self._agent_ids}, "
            f"config={self._env_config})"
        )
