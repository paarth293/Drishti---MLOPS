"""Tests for serialization helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from app.models.schemas import AnomalyPoint
from app.utils.serialization import anomalies_to_payload, datetime_to_iso


def test_datetime_to_iso_returns_iso8601() -> None:
    ts = datetime(2025, 1, 1, 12, 30, tzinfo=UTC)
    assert datetime_to_iso(ts) == "2025-01-01T12:30:00+00:00"


def test_anomalies_to_payload_serialises_points() -> None:
    point = AnomalyPoint(
        ts=datetime(2025, 1, 1, tzinfo=UTC),
        y=42.0,
        y_hat=40.0,
        residual=2.0,
        q_alpha=3.0,
        is_anomaly=True,
    )
    payload = anomalies_to_payload([point])
    assert payload == [
        {
            "ts": "2025-01-01T00:00:00+00:00",
            "y": 42.0,
            "y_hat": 40.0,
            "residual": 2.0,
            "q_alpha": 3.0,
            "is_anomaly": True,
        }
    ]
