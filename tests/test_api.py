"""API integration tests."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

# Configure SQLite test database before importing the app
TEST_DB = Path("data/test_api.db")
TEST_DB.parent.mkdir(exist_ok=True)
if TEST_DB.exists():
    TEST_DB.unlink()
os.environ["DB_URL"] = f"sqlite:///{TEST_DB}"
os.environ["USE_POSTGRES"] = "false"

from app.main import app  # noqa: E402  (import after env configuration)

client = TestClient(app)


def _generate_points(count: int = 80) -> list[dict]:
    base_time = datetime.now(UTC) - timedelta(minutes=count)
    rng = np.random.default_rng(123)
    points = []
    for idx in range(count):
        ts = base_time + timedelta(minutes=idx)
        value = 100 + np.sin(idx / 5) + rng.normal(0, 0.2)
        if idx == count - 5:
            value += 15
        points.append({"ts": ts.isoformat(), "y": float(value)})
    return points


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_detect_and_rca_flow() -> None:
    points = _generate_points()
    ingest_payload = {"service_id": "checkout", "points": points}
    response = client.post("/ingest/series", json=ingest_payload)
    assert response.status_code == 200
    assert response.json()["ingested"] > 0

    detect_payload = {"series": [ingest_payload], "detector": "forecast", "alpha": 0.1}
    response = client.post("/detect/batch", json=detect_payload)
    assert response.status_code == 200
    data = response.json()
    assert data[0]["service_id"] == "checkout"

    rca_response = client.get("/rca/topk", params={"k": 3})
    assert rca_response.status_code == 200
    assert len(rca_response.json()["ranking"]) == 3

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert "aiops_copilot_anomaly_count" in metrics_response.text


def test_graph_endpoint() -> None:
    response = client.get("/graph")
    assert response.status_code == 200
    payload = response.json()
    assert "nodes" in payload and payload["nodes"]
