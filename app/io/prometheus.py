"""Lightweight Prometheus HTTP client."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests


class PrometheusClient:
    """Wraps query operations against a Prometheus-compatible API."""

    def __init__(self, base_url: str, timeout_sec: float = 10.0) -> None:
        """Initialise the client.

        Args:
            base_url: Base URL of the Prometheus HTTP API.
            timeout_sec: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def query_range(self, metric: str, start: datetime, end: datetime, step: str = "60s") -> pd.DataFrame:
        """Fetch a metric time range and return a StatsForecast panel dataframe.

        Args:
            metric: PromQL query string.
            start: Start time for the query window.
            end: End time for the query window.
            step: Step duration between samples (PromQL syntax).

        Returns:
            Dataframe containing columns ``unique_id``, ``ds``, ``y``.

        Raises:
            RuntimeError: When the Prometheus API returns a non-success status.
        """

        params = {
            "query": metric,
            "start": start.astimezone(UTC).isoformat(),
            "end": end.astimezone(UTC).isoformat(),
            "step": step,
        }
        response = requests.get(f"{self.base_url}/api/v1/query_range", params=params, timeout=self.timeout_sec)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        status = payload.get("status")
        if status != "success":
            raise RuntimeError(f"Prometheus query failed with status={status}")

        results = payload.get("data", {}).get("result", [])
        frames = []
        for result in results:
            metric_labels: dict[str, Any] = result.get("metric", {})
            service_id = (
                metric_labels.get("service") or metric_labels.get("service_id") or metric_labels.get("job") or "unknown"
            )
            values = result.get("values", [])
            times = [datetime.fromtimestamp(float(ts), tz=UTC) for ts, _ in values]
            ys = [float(v) for _, v in values]
            frames.append(pd.DataFrame({"unique_id": service_id, "ds": times, "y": ys}))
        if not frames:
            return pd.DataFrame(columns=["unique_id", "ds", "y"])
        return pd.concat(frames, ignore_index=True)
