"""Tests for persistence layer functions."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from importlib import reload

import pytest

from app.io import writers
from app.models.schemas import AnomalyPoint, TimePoint
from app.settings import Settings


@pytest.fixture()
def sqlite_settings(tmp_path) -> Settings:
    db_path = tmp_path / "io_tests.db"
    os.environ["DB_URL"] = f"sqlite:///{db_path}"
    os.environ["USE_POSTGRES"] = "false"
    globals()["writers"] = reload(writers)  # ensure global engine refreshed
    return Settings()


def test_write_and_query_roundtrip(sqlite_settings: Settings) -> None:
    writers.init_db(sqlite_settings)
    service_id = "svc-test"
    base_ts = datetime(2025, 1, 1, tzinfo=UTC)
    measurements = [TimePoint(ts=base_ts + timedelta(minutes=idx), y=float(idx)) for idx in range(3)]
    writers.write_measurements(service_id, measurements)
    stored = writers.query_measurements(service_id, since=None, limit=10)
    assert len(stored) == 3
    anomalies = [
        AnomalyPoint(
            ts=base_ts,
            y=1.0,
            y_hat=0.5,
            residual=0.5,
            q_alpha=0.8,
            is_anomaly=True,
        )
    ]
    writers.write_anomalies(service_id, anomalies)
    retrieved = writers.query_anomalies(service_id, since=None, limit=10)
    assert len(retrieved) == 1
    assert retrieved[0].is_anomaly
