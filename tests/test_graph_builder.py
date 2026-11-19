"""Tests for service graph helpers."""

from __future__ import annotations

import networkx as nx

from app.rca.graph_builder import build_graph_from_config, merge_graphs, normalize_edge_weights


def test_build_and_normalize_graph() -> None:
    config = {
        "nodes": ["checkout", "payment", "inventory"],
        "edges": [
            {"source": "checkout", "target": "payment", "weight": 2.0},
            {"source": "checkout", "target": "inventory", "weight": 1.0},
        ],
    }
    graph = build_graph_from_config(config)
    assert set(graph.nodes) == {"checkout", "payment", "inventory"}
    normalized = normalize_edge_weights(graph)
    weights = [normalized["checkout"][nbr]["weight"] for nbr in normalized.successors("checkout")]
    assert abs(sum(weights) - 1.0) < 1e-6


def test_merge_graphs_accumulates_weights() -> None:
    g1 = nx.DiGraph()
    g1.add_edge("checkout", "payment", weight=0.5)
    g2 = nx.DiGraph()
    g2.add_edge("checkout", "payment", weight=0.7)
    merged = merge_graphs(g1, g2)
    assert merged["checkout"]["payment"]["weight"] == 1.2


def test_normalize_handles_zero_weight() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=0.0)
    normalised = normalize_edge_weights(graph)
    assert normalised["a"]["b"]["weight"] == 0.0
