"""Tests for RCA ranking logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import networkx as nx

from app.rca.graph_builder import normalize_edge_weights
from app.rca.rca_ranker import rca_score


def test_rca_scores_rank_known_cause() -> None:
    graph = nx.DiGraph()
    graph.add_edge("checkout", "payment", weight=1.0)
    graph.add_edge("checkout", "inventory", weight=0.5)
    graph = normalize_edge_weights(graph)
    now = datetime.now(UTC)
    local_scores = {
        "checkout": (2.5, now - timedelta(minutes=5)),
        "payment": (0.5, now - timedelta(minutes=5)),
        "inventory": (0.2, now - timedelta(minutes=5)),
    }
    ranking = rca_score(graph, local_scores, now, alpha=0.6, half_life=timedelta(minutes=10))
    assert ranking[0][0] == "checkout"
    top3 = [node for node, _ in ranking[:3]]
    assert "payment" in top3
