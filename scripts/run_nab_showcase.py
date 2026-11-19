"""End-to-end NAB showcase runner for the AIOps Copilot stack."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests  # noqa: E402

try:
    from scripts.run_evidently_report import generate_report
except ImportError:
    generate_report = None  # type: ignore[assignment]

from scripts.run_nab_ingest import DATA_DIR, ServiceResult, run_ingest_detect  # noqa: E402

SUMMARY_PATH = Path("reports/nab_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NAB ingest, detection, RCA and reporting showcase.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--detector", default="baseline", choices=["forecast", "baseline"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--services", nargs="+", help="Subset of services (defaults to all NAB CSVs).")
    parser.add_argument("--max-workers", type=int, default=0, help="Auto defaults to CPU count when <= 0.")
    parser.add_argument("--anomaly-limit", type=int, default=50, help="Max anomalies to fetch per service.")
    parser.add_argument("--rca-topk", type=int, default=5, help="Top-K RCA ranking to fetch (0 disables).")
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH), help="Where to write the JSON summary.")
    parser.add_argument("--report-service", help="Service id to use for Evidently report (defaults to top anomalies).")
    parser.add_argument("--report-window-ref", type=int, default=500)
    parser.add_argument("--report-window-cur", type=int, default=200)
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip Evidently report generation even when data is available.",
    )
    return parser.parse_args()


def normalise_workers(requested: int) -> int:
    if requested > 0:
        return requested
    return max(1, os.cpu_count() or 1)


def fetch_json(session: requests.Session, url: str, timeout: int, **kwargs: Any) -> Any:
    response = session.get(url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def fetch_text(session: requests.Session, url: str, timeout: int) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def choose_report_service(results: list[ServiceResult], explicit: str | None) -> str | None:
    if explicit:
        return explicit
    ranked = sorted((result for result in results if not result.skipped), key=lambda item: item.anomalies, reverse=True)
    return ranked[0].service_id if ranked and ranked[0].anomalies > 0 else None


def run_showcase(
    *,
    api_url: str,
    data_dir: Path | str = DATA_DIR,
    services: list[str] | None = None,
    detector: str = "baseline",
    alpha: float = 0.1,
    timeout: int = 120,
    chunk_size: int = 500,
    max_workers: int = 0,
    anomaly_limit: int = 50,
    rca_topk: int = 5,
    summary_path: Path | str = SUMMARY_PATH,
    report_service: str | None = None,
    report_window_ref: int = 500,
    report_window_cur: int = 200,
    skip_report: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the NAB showcase end-to-end and persist a summary.

    Returns a dictionary containing the raw results, summary payload,
    summary/report paths and other metadata useful for automation.
    """

    data_dir = Path(data_dir)
    services = services if services else sorted(p.stem for p in data_dir.glob("*.csv"))
    max_workers = normalise_workers(max_workers)
    summary_path = Path(summary_path)

    results = run_ingest_detect(
        api_url=api_url,
        data_dir=data_dir,
        services=services,
        detector=detector,
        alpha=alpha,
        timeout=timeout,
        chunk_size=chunk_size,
        max_workers=max_workers,
    )

    if verbose:
        for result in results:
            print(result.summary())

    summary: dict[str, Any] = {
        "api_url": api_url,
        "detector": detector,
        "alpha": alpha,
        "services": [],
        "rca_topk": [],
        "graph": {},
        "metrics": "",
        "report_path": None,
    }

    with requests.Session() as session:
        for result in results:
            if result.skipped:
                summary["services"].append(
                    {
                        "service_id": result.service_id,
                        "status": "skipped",
                        "message": result.message,
                    }
                )
                continue
            anomalies = fetch_json(
                session,
                f"{api_url}/anomalies",
                timeout=timeout,
                params={"service_id": result.service_id, "limit": anomaly_limit},
            )
            summary["services"].append(
                {
                    "service_id": result.service_id,
                    "ingested": result.ingested,
                    "anomalies_detected": result.anomalies,
                    "anomalies_returned": len(anomalies),
                    "recent_anomalies": anomalies,
                }
            )

        if rca_topk > 0:
            summary["rca_topk"] = fetch_json(
                session,
                f"{api_url}/rca/topk",
                timeout=timeout,
                params={"k": rca_topk},
            )

        summary["graph"] = fetch_json(session, f"{api_url}/graph", timeout=timeout)
        summary["metrics"] = fetch_text(session, f"{api_url}/metrics", timeout=timeout)

    report_path: Path | None = None
    if not skip_report and generate_report is None:
        if verbose:
            print("[warn] Evidently is not installed; skipping report generation.")

    if not skip_report and generate_report is not None:
        chosen_service = choose_report_service(results, report_service)
        if chosen_service:
            try:
                generate_report(chosen_service, report_window_ref, report_window_cur)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[warn] Evidently report generation failed: {exc}")
                summary["report_path"] = None
            else:
                report_path = Path("reports/latest.html")
                summary["report_path"] = str(report_path)
        else:
            summary["report_path"] = None
    else:
        summary["report_path"] = None

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if verbose:
        print(f"\nShowcase summary written to {summary_path}")
        if report_path and report_path.exists():
            print(f"Evidently report available at {report_path}")

    return {
        "results": results,
        "summary": summary,
        "summary_path": summary_path,
        "report_path": report_path,
    }


def main() -> None:
    args = parse_args()
    run_showcase(
        api_url=args.api_url,
        data_dir=args.data_dir,
        services=args.services,
        detector=args.detector,
        alpha=args.alpha,
        timeout=args.timeout,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        anomaly_limit=args.anomaly_limit,
        rca_topk=args.rca_topk,
        summary_path=args.summary_path,
        report_service=args.report_service,
        report_window_ref=args.report_window_ref,
        report_window_cur=args.report_window_cur,
        skip_report=args.skip_report,
        verbose=True,
    )


if __name__ == "__main__":
    main()
