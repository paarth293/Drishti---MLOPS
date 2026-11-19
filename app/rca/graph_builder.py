"""Graph utilities for root cause analysis."""

from __future__ import annotations

import networkx as nx


def build_graph_from_config(config: dict[str, object]) -> nx.DiGraph:
    """Build a directed weighted graph from configuration.

    Args:
        config: Dictionary containing ``nodes`` and ``edges`` definitions.

    Returns:
        Directed graph representing service dependencies.
    """

    graph = nx.DiGraph()
    nodes = config.get("nodes", [])
    edges = config.get("edges", [])
    graph.add_nodes_from(nodes)
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        weight = edge.get("weight", 1.0)
        graph.add_edge(source, target, weight=weight)
    return graph


def merge_graphs(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> nx.DiGraph:
    """Merge two graphs, summing edge weights when duplicates exist.

    Args:
        graph_a: First graph to merge.
        graph_b: Second graph to merge.

    Returns:
        Combined graph with accumulated edge weights.
    """

    merged = nx.DiGraph()
    merged.add_nodes_from(graph_a.nodes)
    merged.add_nodes_from(graph_b.nodes)
    for graph in (graph_a, graph_b):
        for source, target, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            if merged.has_edge(source, target):
                merged[source][target]["weight"] += weight
            else:
                merged.add_edge(source, target, weight=weight)
    return merged


def normalize_edge_weights(graph: nx.DiGraph) -> nx.DiGraph:
    """Normalise outgoing edge weights for each node.

    Args:
        graph: Graph whose edges are to be normalised.

    Returns:
        Graph with outgoing weights normalised to sum to 1.
    """

    normalised = graph.copy()
    for node in normalised.nodes:
        neighbors = list(normalised.successors(node))
        if not neighbors:
            continue
        total_weight = sum(normalised[node][nbr]["weight"] for nbr in neighbors)
        if total_weight <= 0:
            continue
        for neighbor in neighbors:
            normalised[node][neighbor]["weight"] /= total_weight
    return normalised
