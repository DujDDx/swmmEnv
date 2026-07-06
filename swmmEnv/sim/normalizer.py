"""
StateNormalizer - Normalize observations and rewards for RL training.

This module handles standardization of SWMM outputs to ensure
stable and efficient RL training across different model scales.
"""

from typing import Dict, Any, Optional
import numpy as np


class StateNormalizer:
    """
    Normalize observations and rewards for RL training.

    SWMM outputs have inconsistent scales (depth in m, flow in m³/s,
    rainfall in mm/h, flooding volume), which can destabilize training.
    This class provides standardization (z-score) or min-max normalization.

    Example:
        >>> normalizer = StateNormalizer(config)
        >>> normalized_obs = normalizer.normalize_obs(obs, "pump_1")
        >>> normalized_reward = normalizer.normalize_reward(reward)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StateNormalizer.

        Args:
            config: Normalization configuration with structure:
                {
                    'obs': {
                        'depth': {'mean': 2.0, 'std': 1.5},
                        'flow': {'mean': 0.5, 'std': 0.3},
                        'rainfall': {'mean': 5.0, 'std': 10.0},
                    },
                    'reward': {'mean': 0.0, 'std': 10.0}
                }
        """
        self.config = config

        # Observation normalization parameters
        self.obs_params = config.get('obs', {})

        # Reward normalization parameters
        self.reward_params = config.get('reward', {'mean': 0.0, 'std': 1.0})

        # Online statistics update settings
        self.update_online = config.get('update_online', False)
        self.running_stats: Dict[str, Dict[str, float]] = {}

    def normalize_obs(
        self,
        obs: np.ndarray,
        obs_names: Optional[list] = None,
        agent_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Normalize observation array.

        Args:
            obs: Observation array to normalize
            obs_names: List of observation variable names corresponding to obs dimensions.
                      If None, uses default ordering: depth, flow, rainfall, setting
            agent_id: Agent identifier (for per-agent normalization, if configured)

        Returns:
            Normalized observation array
        """
        obs = np.asarray(obs, dtype=np.float32)

        if obs_names is None:
            # Default ordering: depth, flow, rainfall, setting
            obs_names = ['depth', 'flow', 'rainfall', 'setting']

        normalized_obs = np.zeros_like(obs)

        for i, name in enumerate(obs_names):
            if i >= len(obs):
                break

            value = obs[i]
            params = self.obs_params.get(name, {'mean': 0.0, 'std': 1.0})

            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)

            # Z-score normalization
            if std > 0:
                normalized_obs[i] = (value - mean) / std
            else:
                normalized_obs[i] = value - mean

        return normalized_obs

    def normalize_obs_value(self, value: float, obs_name: str) -> float:
        """
        Normalize a single observation value.

        Args:
            value: Observation value
            obs_name: Observation variable name

        Returns:
            Normalized value
        """
        params = self.obs_params.get(obs_name, {'mean': 0.0, 'std': 1.0})
        mean = params.get('mean', 0.0)
        std = params.get('std', 1.0)

        if std > 0:
            return (value - mean) / std
        return value - mean

    def normalize_reward(self, reward: float) -> float:
        """
        Normalize reward value.

        Args:
            reward: Reward value

        Returns:
            Normalized reward
        """
        mean = self.reward_params.get('mean', 0.0)
        std = self.reward_params.get('std', 1.0)

        if std > 0:
            return (reward - mean) / std
        return reward - mean

    def denormalize_reward(self, normalized_reward: float) -> float:
        """
        Convert normalized reward back to original scale.

        Args:
            normalized_reward: Normalized reward value

        Returns:
            Denormalized (original scale) reward
        """
        mean = self.reward_params.get('mean', 0.0)
        std = self.reward_params.get('std', 1.0)

        return normalized_reward * std + mean

    def update_stats(self, obs: np.ndarray, obs_names: Optional[list] = None) -> None:
        """
        Update running statistics for online normalization.

        Args:
            obs: Observation array
            obs_names: List of observation variable names
        """
        if not self.update_online:
            return

        if obs_names is None:
            obs_names = ['depth', 'flow', 'rainfall', 'setting']

        for i, name in enumerate(obs_names):
            if i >= len(obs):
                break

            value = float(obs[i])

            if name not in self.running_stats:
                self.running_stats[name] = {
                    'count': 0,
                    'mean': 0.0,
                    'M2': 0.0  # For Welford's algorithm
                }

            stats = self.running_stats[name]
            count = stats['count'] + 1
            delta = value - stats['mean']
            mean = stats['mean'] + delta / count
            delta2 = value - mean
            M2 = stats['M2'] + delta * delta2

            stats['count'] = count
            stats['mean'] = mean
            stats['M2'] = M2

            # Update std
            if count > 1:
                stats['std'] = np.sqrt(M2 / (count - 1))

    def get_running_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get current running statistics.

        Returns:
            Dictionary of running statistics for each variable
        """
        result = {}
        for name, stats in self.running_stats.items():
            result[name] = {
                'mean': stats['mean'],
                'std': stats.get('std', 1.0),
                'count': stats['count']
            }
        return result

    def set_params_from_running_stats(self) -> None:
        """
        Update normalization parameters from running statistics.

        Call this after collecting sufficient statistics to update
        the normalization parameters.
        """
        for name, stats in self.running_stats.items():
            if stats['count'] > 1:
                self.obs_params[name] = {
                    'mean': stats['mean'],
                    'std': stats.get('std', 1.0)
                }

    def min_max_normalize(
        self,
        value: float,
        min_val: float,
        max_val: float
    ) -> float:
        """
        Apply min-max normalization.

        Args:
            value: Value to normalize
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Normalized value in [0, 1]
        """
        if max_val - min_val > 0:
            return (value - min_val) / (max_val - min_val)
        return 0.5

    def clip_and_normalize(
        self,
        value: float,
        obs_name: str,
        clip_range: Optional[tuple] = None
    ) -> float:
        """
        Clip value to range before normalizing.

        Args:
            value: Value to process
            obs_name: Observation variable name
            clip_range: Optional (min, max) tuple for clipping

        Returns:
            Clipped and normalized value
        """
        if clip_range is not None:
            value = np.clip(value, clip_range[0], clip_range[1])

        return self.normalize_obs_value(value, obs_name)

    def __repr__(self) -> str:
        return f"StateNormalizer(obs_params={self.obs_params}, reward_params={self.reward_params})"