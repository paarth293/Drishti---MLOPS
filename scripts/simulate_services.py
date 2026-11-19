"""Simulate multi-service metrics and push them through the API."""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import requests

API_BASE_URL = "http://localhost:8000"


def build_series(
    service_id: str,
    start: datetime,
    points: int,
    step: timedelta,
    anomaly_rate: float,
) -> list[dict[str, float | str]]:
    """Generate a synthetic sinusoidal series with injected anomalies.

    Args:
        service_id: Service identifier for the generated series.
        start: Initial timestamp.
        points: Number of points to produce.
        step: Time delta between successive points.
        anomaly_rate: Probability of injecting an anomaly at each point.

    Returns:
        List of dictionaries representing time-series measurements.
    """

    timestamps = [start + i * step for i in range(points)]
    rng = np.random.default_rng(seed=42)
    base_frequency = rng.uniform(0.05, 0.15)
    data = []
    for idx, ts in enumerate(timestamps):
        noise = rng.normal(0, 1)
        trend = 0.01 * idx
        value = 50 + 10 * math.sin(base_frequency * idx) + trend + noise
        if rng.random() < anomaly_rate:
            value += rng.choice([-1, 1]) * rng.uniform(15, 30)
        data.append({"ts": ts.isoformat(), "y": float(value)})
    return data


def simulate(args: argparse.Namespace) -> None:
    """Run the simulation workflow.

    Args:
        args: Parsed command-line arguments.
    """

    start_time = datetime.now(UTC) - timedelta(minutes=args.duration_min)
    step = timedelta(seconds=args.freq_s)
    points = int(args.duration_min * 60 / args.freq_s)

    for service_idx in range(args.nb_services):
        service_id = f"service_{service_idx}"
        series = build_series(service_id, start_time, points, step, args.anomaly_rate)
        response = requests.post(
            f"{API_BASE_URL}/ingest/series",
            json={"service_id": service_id, "points": series},
            timeout=30,
        )
        response.raise_for_status()

    detect_payload = {
        "series": [
            {
                "service_id": f"service_{idx}",
                "points": build_series(f"service_{idx}", start_time, points, step, args.anomaly_rate),
            }
            for idx in range(args.nb_services)
        ],
        "detector": args.detector,
        "alpha": args.alpha,
    }
    response = requests.post(f"{API_BASE_URL}/detect/batch", json=detect_payload, timeout=60)
    response.raise_for_status()
    print("Detection completed:", response.json())


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace with simulation parameters.
    """

    parser = argparse.ArgumentParser(description="Simulate services for AIOps Copilot.")
    parser.add_argument("--nb-services", type=int, default=5)
    parser.add_argument("--duration-min", type=float, default=120)
    parser.add_argument("--freq-s", type=int, default=60)
    parser.add_argument("--anomaly-rate", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--detector", type=str, default="forecast", choices=["forecast", "baseline"])
    return parser.parse_args()


if __name__ == "__main__":
    simulate(parse_args())
