"""
Feature extractors for observation construction.

Each entry in FEATURE_EXTRACTOR_REGISTRY maps a feature name string
to a tuple of (extractor_fn, normalization_name).

Extractor signature: (engine, agent_config, obs_raingage) -> float
"""

from typing import Dict, Any, Tuple, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from swmmEnv.sim.engine import SWMMEngine


# Feature extractor registry: name -> (extractor_fn, norm_name)
FEATURE_EXTRACTOR_REGISTRY: Dict[str, Tuple[Callable, str]] = {}


def _extract_upstream_depth(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract upstream node depth."""
    upstream_node = agent_config.get('upstream_node')
    if upstream_node:
        try:
            return engine.get_node_state(upstream_node)['depth']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_downstream_depth(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract downstream node depth."""
    downstream_node = agent_config.get('downstream_node')
    if downstream_node:
        try:
            return engine.get_node_state(downstream_node)['depth']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_flow(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract link flow rate."""
    link_id = agent_config.get('link_id')
    if link_id:
        try:
            return engine.get_link_state(link_id)['flow']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_setting(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract current control setting."""
    link_id = agent_config.get('link_id')
    if link_id:
        try:
            return engine.get_link_state(link_id)['current_setting']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_rainfall(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract rainfall from configured raingage."""
    try:
        return engine.get_rainfall(obs_raingage)
    except (KeyError, ValueError):
        return 0.0


def _extract_total_flooding(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract total flooding across all nodes."""
    try:
        return engine.get_total_flooding()
    except (KeyError, ValueError):
        return 0.0


def _extract_upstream_flooding(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract upstream node flooding rate."""
    upstream_node = agent_config.get('upstream_node')
    if upstream_node:
        try:
            return engine.get_node_state(upstream_node)['flooding']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_downstream_flooding(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract downstream node flooding rate."""
    downstream_node = agent_config.get('downstream_node')
    if downstream_node:
        try:
            return engine.get_node_state(downstream_node)['flooding']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_upstream_head(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract upstream node water elevation (head)."""
    upstream_node = agent_config.get('upstream_node')
    if upstream_node:
        try:
            return engine.get_node_state(upstream_node)['head']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_downstream_head(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract downstream node water elevation (head)."""
    downstream_node = agent_config.get('downstream_node')
    if downstream_node:
        try:
            return engine.get_node_state(downstream_node)['head']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_upstream_volume(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract upstream node stored volume."""
    upstream_node = agent_config.get('upstream_node')
    if upstream_node:
        try:
            return engine.get_node_state(upstream_node)['volume']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_downstream_volume(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract downstream node stored volume."""
    downstream_node = agent_config.get('downstream_node')
    if downstream_node:
        try:
            return engine.get_node_state(downstream_node)['volume']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_upstream_inflow(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract upstream node total inflow rate."""
    upstream_node = agent_config.get('upstream_node')
    if upstream_node:
        try:
            return engine.get_node_state(upstream_node)['total_inflow']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_downstream_inflow(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract downstream node total inflow rate."""
    downstream_node = agent_config.get('downstream_node')
    if downstream_node:
        try:
            return engine.get_node_state(downstream_node)['total_inflow']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_link_depth(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract link flow depth."""
    link_id = agent_config.get('link_id')
    if link_id:
        try:
            return engine.get_link_state(link_id)['depth']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


def _extract_link_volume(
    engine: 'SWMMEngine',
    agent_config: Dict[str, Any],
    obs_raingage: Optional[str]
) -> float:
    """Extract link flow volume."""
    link_id = agent_config.get('link_id')
    if link_id:
        try:
            return engine.get_link_state(link_id)['volume']
        except (KeyError, ValueError):
            return 0.0
    return 0.0


# Populate registry with built-in extractors
FEATURE_EXTRACTOR_REGISTRY['upstream_depth'] = (_extract_upstream_depth, 'depth')
FEATURE_EXTRACTOR_REGISTRY['downstream_depth'] = (_extract_downstream_depth, 'depth')
FEATURE_EXTRACTOR_REGISTRY['flow'] = (_extract_flow, 'flow')
FEATURE_EXTRACTOR_REGISTRY['setting'] = (_extract_setting, 'setting')
FEATURE_EXTRACTOR_REGISTRY['rainfall'] = (_extract_rainfall, 'rainfall')
FEATURE_EXTRACTOR_REGISTRY['total_flooding'] = (_extract_total_flooding, 'flooding')
FEATURE_EXTRACTOR_REGISTRY['upstream_flooding'] = (_extract_upstream_flooding, 'flooding')
FEATURE_EXTRACTOR_REGISTRY['downstream_flooding'] = (_extract_downstream_flooding, 'flooding')
FEATURE_EXTRACTOR_REGISTRY['upstream_head'] = (_extract_upstream_head, 'depth')
FEATURE_EXTRACTOR_REGISTRY['downstream_head'] = (_extract_downstream_head, 'depth')
FEATURE_EXTRACTOR_REGISTRY['upstream_volume'] = (_extract_upstream_volume, 'volume')
FEATURE_EXTRACTOR_REGISTRY['downstream_volume'] = (_extract_downstream_volume, 'volume')
FEATURE_EXTRACTOR_REGISTRY['upstream_inflow'] = (_extract_upstream_inflow, 'flow')
FEATURE_EXTRACTOR_REGISTRY['downstream_inflow'] = (_extract_downstream_inflow, 'flow')
FEATURE_EXTRACTOR_REGISTRY['link_depth'] = (_extract_link_depth, 'depth')
FEATURE_EXTRACTOR_REGISTRY['link_volume'] = (_extract_link_volume, 'volume')


def get_feature_extractor(name: str) -> Tuple[Callable, str]:
    """
    Get feature extractor and normalization name.

    Args:
        name: Feature name

    Returns:
        Tuple of (extractor_fn, norm_name)

    Raises:
        KeyError: If feature name not found
    """
    if name not in FEATURE_EXTRACTOR_REGISTRY:
        available = list(FEATURE_EXTRACTOR_REGISTRY.keys())
        raise KeyError(
            f"Unknown feature '{name}'. Available features: {available}"
        )
    return FEATURE_EXTRACTOR_REGISTRY[name]


def register_feature_extractor(
    name: str,
    extractor_fn: Callable,
    norm_name: str
) -> None:
    """
    Register a custom feature extractor.

    Args:
        name: Feature name
        extractor_fn: Extractor function with signature
            (engine, agent_config, obs_raingage) -> float
        norm_name: Normalization variable name for z-score normalization

    Example:
        >>> def my_depth_extractor(engine, agent_config, obs_raingage):
        ...     node_id = agent_config.get('custom_node')
        ...     return engine.get_node_state(node_id)['depth'] if node_id else 0.0
        >>> register_feature_extractor('custom_depth', my_depth_extractor, 'depth')
    """
    if name in FEATURE_EXTRACTOR_REGISTRY:
        raise ValueError(f"Feature '{name}' already registered")
    FEATURE_EXTRACTOR_REGISTRY[name] = (extractor_fn, norm_name)


def list_available_features() -> list:
    """
    List all available feature names.

    Returns:
        List of registered feature names
    """
    return list(FEATURE_EXTRACTOR_REGISTRY.keys())


__all__ = [
    'FEATURE_EXTRACTOR_REGISTRY',
    'get_feature_extractor',
    'register_feature_extractor',
    'list_available_features',
]