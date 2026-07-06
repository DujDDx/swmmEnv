"""
Default reward function for SWMMEnv.

This module provides a default reward function that encourages:
- Minimizing flooding
- Maintaining target water levels
- Energy efficiency
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from swmmEnv.sim.engine import SWMMEngine


def default_reward(engine: 'SWMMEngine', config: Dict[str, Any]) -> float:
    """
    Compute default reward for SWMM control.

    Reward components:
    1. Flooding penalty: -weight * total_flooding
    2. Level deviation penalty: -weight * sum(|depth - target|)
    3. Energy penalty (optional): -weight * pump_energy

    Args:
        engine: SWMMEngine instance for state retrieval
        config: Configuration dictionary containing:
            - reward_weights: Dict of weights for each component
            - target_levels: Dict of node_id -> target depth
            - max_depth: Maximum depth for normalization (optional)

    Returns:
        Reward value (typically negative, higher is better)
    """
    reward = 0.0

    # Get reward weights
    weights = config.get('reward_weights', {})
    flooding_weight = weights.get('flooding', 1.0)
    level_weight = weights.get('level_deviation', 0.5)
    energy_weight = weights.get('energy', 0.2)

    # 1. Flooding penalty
    total_flooding = engine.get_total_flooding()
    reward -= flooding_weight * total_flooding

    # 2. Level deviation penalty
    target_levels = config.get('target_levels', {})

    if target_levels:
        total_deviation = 0.0

        for node_id, target_depth in target_levels.items():
            try:
                node_state = engine.get_node_state(node_id)
                current_depth = node_state['depth']
                deviation = abs(current_depth - target_depth)
                total_deviation += deviation
            except (KeyError, ValueError):
                # Node not found, skip
                continue

        reward -= level_weight * total_deviation

    # 3. Energy efficiency penalty (based on pump settings)
    # Higher pump settings = more energy consumption
    agents = config.get('agents', {})

    if energy_weight > 0:
        total_energy = 0.0

        for agent_id, agent_config in agents.items():
            if agent_config.get('type') == 'pump':
                try:
                    link_state = engine.get_link_state(agent_config['link_id'])
                    # Energy proportional to flow rate (simplified)
                    flow = link_state['flow']
                    setting = link_state['current_setting']

                    # Energy = flow * setting (simplified model)
                    if flow > 0:
                        energy = abs(flow) * setting
                        total_energy += energy
                except (KeyError, ValueError):
                    continue

        reward -= energy_weight * total_energy

    return reward


def flooding_only_reward(engine: 'SWMMEngine', config: Dict[str, Any]) -> float:
    """
    Simple reward based only on flooding minimization.

    Args:
        engine: SWMMEngine instance
        config: Configuration dictionary

    Returns:
        Negative flooding value
    """
    total_flooding = engine.get_total_flooding()
    return -total_flooding


def normalized_flooding_reward(
    engine: 'SWMMEngine',
    config: Dict[str, Any]
) -> float:
    """
    Normalized flooding reward in range [-1, 0].

    Args:
        engine: SWMMEngine instance
        config: Configuration dictionary with 'max_flooding' parameter

    Returns:
        Normalized reward between -1 and 0
    """
    total_flooding = engine.get_total_flooding()
    max_flooding = config.get('max_flooding', 10.0)  # m³/s

    # Normalize to [-1, 0]
    normalized = -min(total_flooding / max_flooding, 1.0)

    return normalized


def multi_objective_reward(
    engine: 'SWMMEngine',
    config: Dict[str, Any]
) -> float:
    """
    Multi-objective reward combining flooding, level, and stability.

    Args:
        engine: SWMMEngine instance
        config: Configuration dictionary

    Returns:
        Combined reward value
    """
    reward = 0.0

    # Flooding component
    total_flooding = engine.get_total_flooding()
    reward -= 10.0 * total_flooding  # Scale up flooding importance

    # Level tracking
    target_levels = config.get('target_levels', {})

    for node_id, target_depth in target_levels.items():
        try:
            node_state = engine.get_node_state(node_id)
            deviation = abs(node_state['depth'] - target_depth)
            reward -= deviation
        except (KeyError, ValueError):
            continue

    # Stability bonus (penalize large setting changes)
    # This would require tracking previous settings
    # For simplicity, skipped in this implementation

    return reward


# Reward function registry
REWARD_REGISTRY = {
    'default_reward': default_reward,
    'flooding_only': flooding_only_reward,
    'normalized_flooding': normalized_flooding_reward,
    'multi_objective': multi_objective_reward,
}


def get_reward_fn(name: str):
    """
    Get reward function by name.

    Args:
        name: Reward function name

    Returns:
        Reward function callable

    Raises:
        KeyError: If reward function not found
    """
    if name not in REWARD_REGISTRY:
        available = list(REWARD_REGISTRY.keys())
        raise KeyError(
            f"Unknown reward function: {name}. "
            f"Available: {available}"
        )

    return REWARD_REGISTRY[name]