"""Tempo client for optional OpenTelemetry trace ingestion."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import httpx
import networkx as nx

LOGGER = logging.getLogger(__name__)


class TempoClient:
    """Fetch traces from Tempo and build service dependency graphs."""

    def __init__(self, base_url: str, enabled: bool = False, timeout_sec: float = 5.0) -> None:
        """Initialise the Tempo client.

        Args:
            base_url: Base URL of the Tempo API.
            enabled: Whether the client is active.
            timeout_sec: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.timeout_sec = timeout_sec

    def fetch_recent_traces(self, window_min: int) -> list[dict[str, object]]:
        """Return recent traces within the configured time window.

        Args:
            window_min: Time window size in minutes.

        Returns:
            List of trace dictionaries returned by Tempo.
        """

        if not self.enabled:
            return []
        end = datetime.now(UTC)
        start = end - timedelta(minutes=window_min)
        params = {
            "q": "",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 200,
        }
        try:
            response = httpx.get(f"{self.base_url}/api/search", params=params, timeout=self.timeout_sec)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.warning("tempo_fetch_failed", extra={"message": str(exc)})
            return []
        payload = response.json()
        traces = payload.get("traces") or payload.get("data") or []
        return traces

    @staticmethod
    def build_service_graph_from_traces(traces: list[dict[str, object]]) -> nx.DiGraph:
        """Construct a dependency graph from Tempo trace payloads.

        Args:
            traces: Collection of trace documents.

        Returns:
            Directed graph where edges represent parent-child relationships.
        """

        graph = nx.DiGraph()
        for trace in traces:
            spans = trace.get("spans") or []
            span_service: dict[str, str] = {}
            for span in spans:
                span_id = span.get("spanID") or span.get("span_id")
                process = span.get("process", {})
                service = process.get("serviceName") or span.get("service", "unknown")
                if span_id is not None:
                    span_service[span_id] = service
                graph.add_node(service)
            for span in spans:
                child_service = span_service.get(span.get("spanID"))
                if child_service is None:
                    continue
                references = span.get("references") or []
                for ref in references:
                    if ref.get("refType") not in {"CHILD_OF", "FOLLOWS_FROM"}:
                        continue
                    parent_service = span_service.get(ref.get("spanID"))
                    if parent_service and parent_service != child_service:
                        previous = graph.get_edge_data(parent_service, child_service, {}).get("weight", 0.0)
                        graph.add_edge(parent_service, child_service, weight=previous + 1.0)
        return graph
