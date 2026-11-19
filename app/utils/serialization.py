"""Serialization helpers for API responses and storage."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from app.models.schemas import AnomalyPoint


def datetime_to_iso(value: datetime) -> str:
    """Serialise datetime as ISO8601 string.

    Args:
        value: Datetime to serialise.

    Returns:
        ISO8601 string representation.
    """

    return value.isoformat()


def anomalies_to_payload(points: Iterable[AnomalyPoint]) -> list[dict]:
    """Convert anomaly points to JSON-serialisable dictionaries.

    Args:
        points: Anomaly objects to convert.

    Returns:
        List of dictionaries ready for JSON encoding.
    """

    payload: list[dict] = []
    for point in points:
        payload.append(
            {
                "ts": datetime_to_iso(point.ts),
                "y": point.y,
                "y_hat": point.y_hat,
                "residual": point.residual,
                "q_alpha": point.q_alpha,
                "is_anomaly": point.is_anomaly,
            }
        )
    return payload
