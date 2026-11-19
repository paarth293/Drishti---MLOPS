"""Ingest and detect anomalies on NAB series."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data/nab")


@dataclass
class ServiceResult:
    """Summary of ingestion and detection for a single service."""

    service_id: str
    ingested: int
    anomalies: int
    detector: str
    skipped: bool = False
    message: str | None = None

    def summary(self) -> str:
        if self.skipped:
            return self.message or f"{self.service_id}: skipped"
        return f"{self.service_id}: ingested={self.ingested}, anomalies={self.anomalies}, detector={self.detector}"


def load_series(path: Path) -> list[dict]:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "y" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value": "y"})
    if "y" not in df.columns:
        missing = ", ".join(df.columns)
        raise KeyError(f"Column 'y' missing in {path} (available: {missing})")
    return [{"ts": ts.isoformat(), "y": float(val)} for ts, val in zip(df["timestamp"], df["y"], strict=False)]


def chunk(points: list[dict], size: int) -> Iterable[list[dict]]:
    if size <= 0:
        yield points
    else:
        for idx in range(0, len(points), size):
            yield points[idx : idx + size]


def ingest(session: requests.Session, api_url: str, service_id: str, points: list[dict], timeout: int) -> None:
    response = session.post(
        f"{api_url}/ingest/series",
        json={"service_id": service_id, "points": points},
        timeout=timeout,
    )
    response.raise_for_status()


def detect(
    session: requests.Session,
    api_url: str,
    service_id: str,
    points: list[dict],
    detector: str,
    alpha: float,
    timeout: int,
    chunk_size: int,
) -> int:
    anomalies_total = 0
    for window in chunk(points, chunk_size):
        payload = {"series": [{"service_id": service_id, "points": window}], "detector": detector, "alpha": alpha}
        response = session.post(f"{api_url}/detect/batch", json=payload, timeout=timeout)
        response.raise_for_status()
        anomalies_total += len(response.json()[0]["anomalies"])
    return anomalies_total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NAB CSVs and run detection.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--detector", default="baseline", choices=["forecast", "baseline"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=500, help="Points per detection request (0 = all).")
    parser.add_argument(
        "--services",
        nargs="+",
        help="Subset of service IDs to process (defaults to all in data directory).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Maximum parallel workers to use across services (default: CPU count).",
    )
    return parser.parse_args()


def process_service(
    api_url: str,
    data_dir: Path,
    service: str,
    detector: str,
    alpha: float,
    timeout: int,
    chunk_size: int,
) -> ServiceResult:
    csv_path = data_dir / f"{service}.csv"
    if not csv_path.exists():
        return ServiceResult(
            service_id=service,
            ingested=0,
            anomalies=0,
            detector=detector,
            skipped=True,
            message=f"Skipping missing series: {csv_path}",
        )

    points = load_series(csv_path)
    with requests.Session() as session:
        ingest(session, api_url, service, points, timeout)
        anomalies = detect(session, api_url, service, points, detector, alpha, timeout, chunk_size)
    return ServiceResult(service_id=service, ingested=len(points), anomalies=anomalies, detector=detector)


def run_ingest_detect(
    api_url: str,
    data_dir: Path,
    services: list[str],
    detector: str,
    alpha: float,
    timeout: int,
    chunk_size: int,
    max_workers: int,
) -> list[ServiceResult]:
    if not services:
        raise RuntimeError("No services to process. Provide --services or populate the data directory.")

    max_workers = max(1, min(max_workers, len(services)))
    results: list[ServiceResult] = []
    failures: dict[str, Exception] = {}

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_service = {
            executor.submit(
                process_service,
                api_url,
                data_dir,
                service,
                detector,
                alpha,
                timeout,
                chunk_size,
            ): service
            for service in services
        }

        for future in cf.as_completed(future_to_service):
            service = future_to_service[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - surface worker failure
                failures[service] = exc
            else:
                results.append(result)

    if failures:
        joined = "; ".join(f"{svc}: {err}" for svc, err in failures.items())
        raise RuntimeError(f"Ingestion/detection failed for services: {joined}")

    # Preserve consistent ordering with input list for determinism.
    order_index = {service: idx for idx, service in enumerate(services)}
    results.sort(key=lambda item: order_index.get(item.service_id, len(services)))
    return results


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    services = args.services if args.services else sorted(p.stem for p in data_dir.glob("*.csv"))

    try:
        results = run_ingest_detect(
            api_url=args.api_url,
            data_dir=data_dir,
            services=services,
            detector=args.detector,
            alpha=args.alpha,
            timeout=args.timeout,
            chunk_size=args.chunk_size,
            max_workers=args.max_workers,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    for result in results:
        print(result.summary())


if __name__ == "__main__":
    main()
