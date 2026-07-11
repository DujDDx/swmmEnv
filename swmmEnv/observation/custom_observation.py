"""
Custom observation function template for SWMMEnv.

Use this module as a template for creating custom observation functions.
"""

from typing import Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from swmmEnv.sim.engine import SWMMEngine


class CustomObservationFunction:
    """
    Base class for custom observation functions.

    Override __call__ to implement your own observation logic.

    Example:
        >>> class MyObservation(CustomObservationFunction):
        ...     def __init__(self):
        ...         self.cache = {}
        ...
        ...     def __call__(self, engine, agent_id, config):
        ...         agent_cfg = config['agents'][agent_id]
        ...         upstream = agent_cfg['upstream_node']
        ...         depth = engine.get_node_state(upstream)['depth']
        ...         flow = engine.get_link_state(agent_cfg['link_id'])['flow']
        ...         return np.array([depth, flow], dtype=np.float32)
        ...
        ...     def get_obs_dim(self) -> int:
        ...         return 2
    """

    def __call__(
        self,
        engine: 'SWMMEngine',
        agent_id: str,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute observation for an agent.

        Args:
            engine: SWMMEngine instance for state retrieval
            agent_id: Agent identifier
            config: Configuration dictionary

        Returns:
            Observation array (np.ndarray)

        Raises:
            NotImplementedError: Subclass must implement
        """
        raise NotImplementedError(
            "Subclass must implement __call__(engine, agent_id, config)"
        )

    def get_obs_dim(self) -> int:
        """
        Return observation dimension.

        Override this method to specify the dimension of observations
        produced by this function.

        Returns:
            Observation dimension
        """
        raise NotImplementedError(
            "Subclass must implement get_obs_dim()"
        )

    def reset(self) -> None:
        """
        Reset any internal state.

        Override this method if your observation function maintains
        state across steps (e.g., for history, smoothing, etc.).
        """
        pass


# Example: History-aware observation function
class HistoryObservationFunction(CustomObservationFunction):
    """
    Observation function that includes recent history.

    This is an example showing how to maintain state
    across observation calls.
    """

    def __init__(self, history_length: int = 3, base_features: list = None):
        """
        Initialize history observation function.

        Args:
            history_length: Number of historical steps to include
            base_features: List of feature names to extract
        """
        self.history_length = history_length
        self.base_features = base_features or ['upstream_depth', 'flow', 'setting']
        self.history: Dict[str, list] = {}

    def __call__(
        self,
        engine: 'SWMMEngine',
        agent_id: str,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Compute observation with history."""
        from swmmEnv.observation.feature_extractors import get_feature_extractor

        agent_config = config['agents'][agent_id]
        obs_raingage = config.get('obs_raingage')

        # Get current features
        current_obs = []
        for feature_name in self.base_features:
            extractor_fn, _ = get_feature_extractor(feature_name)
            value = extractor_fn(engine, agent_config, obs_raingage)
            current_obs.append(value)

        # Update history
        if agent_id not in self.history:
            self.history[agent_id] = []
        self.history[agent_id].append(current_obs)

        # Keep only recent history
        if len(self.history[agent_id]) > self.history_length:
            self.history[agent_id] = self.history[agent_id][-self.history_length:]

        # Flatten history into observation
        obs = []
        for hist_obs in self.history[agent_id]:
            obs.extend(hist_obs)

        return np.array(obs, dtype=np.float32)

    def get_obs_dim(self) -> int:
        """Return observation dimension."""
        return len(self.base_features) * self.history_length

    def reset(self) -> None:
        """Reset history."""
        self.history = {}


__all__ = [
    'CustomObservationFunction',
    'HistoryObservationFunction',
]