"""
Custom reward function template for SWMMEnv.

Use this module as a template for creating custom reward functions.
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from swmmEnv.sim.engine import SWMMEngine


def custom_reward_template(engine: 'SWMMEngine', config: Dict[str, Any]) -> float:
    """
    Custom reward function template.

    Override this function to implement your own reward logic.

    Args:
        engine: SWMMEngine instance for retrieving simulation state
        config: Configuration dictionary

    Returns:
        Reward value (float)

    Example implementation:
        >>> def my_reward(engine, config):
        ...     # Get state
        ...     flooding = engine.get_total_flooding()
        ...     node_state = engine.get_node_state("J1")
        ...
        ...     # Compute reward
        ...     reward = -flooding
        ...     if node_state['depth'] < 1.0:
        ...         reward += 1.0  # Bonus for low depth
        ...
        ...     return reward
    """
    # Default: negative flooding
    total_flooding = engine.get_total_flooding()

    return -total_flooding


class CustomRewardFunction:
    """
    Class-based custom reward function.

    Use this when you need to maintain state between reward computations,
    such as tracking previous states for stability rewards.

    Example:
        >>> class StabilityReward(CustomRewardFunction):
        ...     def __init__(self):
        ...         self.previous_depths = {}
        ...
        ...     def __call__(self, engine, config):
        ...         reward = -engine.get_total_flooding()
        ...
        ...         # Stability bonus
        ...         for node_id in config.get('obs_nodes', []):
        ...             depth = engine.get_node_state(node_id)['depth']
        ...             if node_id in self.previous_depths:
        ...                 change = abs(depth - self.previous_depths[node_id])
        ...                 reward -= change
        ...             self.previous_depths[node_id] = depth
        ...
        ...         return reward
    """

    def __init__(self, **kwargs):
        """
        Initialize custom reward function.

        Args:
            **kwargs: Custom parameters for reward computation
        """
        self.params = kwargs

    def __call__(self, engine: 'SWMMEngine', config: Dict[str, Any]) -> float:
        """
        Compute reward.

        Args:
            engine: SWMMEngine instance
            config: Configuration dictionary

        Returns:
            Reward value
        """
        raise NotImplementedError("Subclass must implement __call__")


# Example: Stability-aware reward
class StabilityReward(CustomRewardFunction):
    """
    Reward function that penalizes rapid state changes.

    Encourages smooth control policies.
    """

    def __init__(self, stability_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.stability_weight = stability_weight
        self.previous_depths: Dict[str, float] = {}

    def __call__(self, engine: 'SWMMEngine', config: Dict[str, Any]) -> float:
        reward = -engine.get_total_flooding()

        # Stability component
        obs_nodes = config.get('obs_nodes', [])

        for node_id in obs_nodes:
            try:
                depth = engine.get_node_state(node_id)['depth']

                if node_id in self.previous_depths:
                    change = abs(depth - self.previous_depths[node_id])
                    reward -= self.stability_weight * change

                self.previous_depths[node_id] = depth
            except (KeyError, ValueError):
                continue

        return reward

    def reset(self) -> None:
        """Reset state tracking."""
        self.previous_depths = {}