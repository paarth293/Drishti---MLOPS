"""Root cause analysis scoring based on graph propagation."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import networkx as nx


def _decay(delta: timedelta, half_life: timedelta) -> float:
    """Exponential decay controlled by half-life.

    Args:
        delta: Time difference since the anomaly occurred.
        half_life: Half-life duration controlling decay.

    Returns:
        Decay multiplier between 0 and 1.
    """

    if half_life.total_seconds() <= 0:
        return 1.0
    return math.exp(-math.log(2) * delta.total_seconds() / half_life.total_seconds())


def _local_intensity(local_scores: dict[str, tuple[float, datetime]], service_id: str, now: datetime) -> float:
    """Return local intensity capped to reduce outliers.

    Args:
        local_scores: Mapping of services to their intensity and timestamp.
        service_id: Service of interest.
        now: Current reference time.

    Returns:
        Capped local intensity value.
    """

    if service_id not in local_scores:
        return 0.0
    score, timestamp = local_scores[service_id]
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    delta = now - timestamp
    if delta.total_seconds() < 0:
        delta = timedelta(0)
    return min(max(score, 0.0), 3.0)


def rca_score(
    graph: nx.DiGraph,
    local_scores: dict[str, tuple[float, datetime]],
    now: datetime,
    alpha: float,
    half_life: timedelta,
) -> list[tuple[str, float]]:
    """Compute RCA scores based on graph topology and local anomaly intensities.

    Args:
        graph: Dependency graph.
        local_scores: Map from service id to (intensity, timestamp).
        now: Current evaluation timestamp.
        alpha: Weight for local vs propagated scores.
        half_life: Half-life applied to propagation decay.

    Returns:
        Ranked list of services with RCA scores in descending order.
    """

    rankings: list[tuple[str, float]] = []
    nodes = list(graph.nodes)
    for node in nodes:
        local = _local_intensity(local_scores, node, now)
        propagated = 0.0
        for predecessor in graph.predecessors(node):
            neighbor_local = _local_intensity(local_scores, predecessor, now)
            if neighbor_local == 0.0:
                continue
            weight = graph[predecessor][node].get("weight", 1.0)
            neighbor_timestamp = local_scores.get(predecessor, (0.0, now))[1]
            if neighbor_timestamp.tzinfo is None:
                neighbor_timestamp = neighbor_timestamp.replace(tzinfo=UTC)
            decay_factor = _decay(now - neighbor_timestamp, half_life)
            propagated += weight * neighbor_local * decay_factor
        score = alpha * local + (1 - alpha) * propagated
        rankings.append((node, score))
    rankings.sort(key=lambda item: item[1], reverse=True)
    return rankings
