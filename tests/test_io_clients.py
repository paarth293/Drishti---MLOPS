"""Tests for Prometheus and Tempo clients."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.io.otel_tempo import TempoClient
from app.io.prometheus import PrometheusClient


class _MockResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401 - mimic requests API
        return

    def json(self) -> dict[str, Any]:
        return self._payload


def test_prometheus_query_range(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"service": "checkout"},
                    "values": [
                        (datetime(2025, 1, 1, tzinfo=UTC).timestamp(), "1"),
                        (datetime(2025, 1, 1, 1, tzinfo=UTC).timestamp(), "2"),
                    ],
                }
            ]
        },
    }

    def _fake_get(url: str, params: dict[str, Any], timeout: float) -> _MockResponse:  # noqa: ARG001
        return _MockResponse(payload)

    monkeypatch.setattr("app.io.prometheus.requests.get", _fake_get)
    client = PrometheusClient("http://prometheus")
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    df = client.query_range("dummy", start=start, end=end)
    assert not df.empty
    assert df.iloc[0]["unique_id"] == "checkout"


def test_tempo_build_service_graph_from_traces() -> None:
    traces = [
        {
            "spans": [
                {
                    "spanID": "1",
                    "process": {"serviceName": "checkout"},
                },
                {
                    "spanID": "2",
                    "process": {"serviceName": "payment"},
                    "references": [{"refType": "CHILD_OF", "spanID": "1"}],
                },
            ]
        }
    ]
    graph = TempoClient.build_service_graph_from_traces(traces)
    assert graph.has_edge("checkout", "payment")
