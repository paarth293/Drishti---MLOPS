"""Apply a graph configuration to the running API."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import requests
import yaml


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {
        "nodes": data.get("nodes", []),
        "edges": data.get("edges", []),
    }


def apply_graph(api_url: str, payload: dict[str, Any]) -> None:
    response = requests.post(f"{api_url}/graph", json=payload, timeout=30)
    response.raise_for_status()
    print(f"Applied graph with {len(payload['nodes'])} nodes / {len(payload['edges'])} edges.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply graph configuration to API.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--graph", required=True, help="Path to graph YAML file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    graph_path = Path(args.graph)
    graph_payload = load_graph(graph_path)
    apply_graph(args.api_url, graph_payload)
