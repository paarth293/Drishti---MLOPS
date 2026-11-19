"""Drishti Evidently Data-Drift Report + Auto-Retraining Feedback Loop.

Upgrades the original Evidently report script to:
1. Compute reference vs. current data drift from SQLite/Postgres measurements.
2. Save `reports/latest.html` and `reports/drift_summary.json`.
3. Automatically trigger MLflow recalibration/retraining when `dataset_drift` is True
   (or when `--force-retrain` is passed for live viva demonstrations).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import create_engine, text

from app.settings import load_settings


def load_measurements(service_id: str, limit: int) -> pd.DataFrame:
    """Load the latest `limit` measurements for a service ordered chronologically."""
    settings = load_settings()
    engine = create_engine(settings.effective_db_url)
    query = text(
        "SELECT ts, y FROM measurements WHERE service_id = :service_id "
        "ORDER BY ts DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"service_id": service_id, "limit": limit})
    if df.empty:
        return df
    return df.sort_values("ts").reset_index(drop=True)


def _extract_drift_flags(report_dict: dict[str, Any]) -> tuple[bool, float]:
    """Safely extract (dataset_drift, drift_share) from Evidently's report dict."""
    for metric in report_dict.get("metrics", []):
        res = metric.get("result", {})
        if "dataset_drift" in res:
            return bool(res.get("dataset_drift", False)), float(res.get("share_of_drifted_columns", 0.0))
    return False, 0.0


def generate_report(
    service_id: str,
    window_ref: int = 500,
    window_cur: int = 200,
    auto_retrain: bool = True,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """Generate Evidently HTML + JSON drift report and trigger retraining if drift is detected."""
    df = load_measurements(service_id, window_ref + window_cur)
    if len(df) < 10:
        raise ValueError(
            f"Not enough measurements for service '{service_id}' (found {len(df)}, need >= 10)."
        )

    split_idx = min(window_ref, max(len(df) // 2, 1))
    ref_df = df.iloc[:split_idx][["y"]].copy()
    cur_df = df.iloc[split_idx : split_idx + window_cur][["y"]].copy()
    if cur_df.empty:
        cur_df = df.iloc[split_idx:][["y"]].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "latest.html"
    report.save_html(str(html_path))

    drift_detected, drift_share = _extract_drift_flags(report.as_dict())
    retrain_result = None

    if auto_retrain and (drift_detected or force_retrain):
        reason = "drift_detected_auto_retrain" if drift_detected else "forced_drift_demo_retrain"
        print(
            f"[DRISHTI DRIFT TRIGGER] service={service_id} "
            f"dataset_drift={drift_detected} drift_share={drift_share:.2f} -> triggering MLflow retraining..."
        )
        try:
            from scripts.mlflow_tracking import run_mlflow_pipeline

            retrain_result = run_mlflow_pipeline(
                service_id=service_id,
                promote_champion=True,
                trigger_reason=reason,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[DRISHTI DRIFT TRIGGER] MLflow retrain skipped: {exc}")

    summary = {
        "project": "Drishti",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "service_id": service_id,
        "window_ref": len(ref_df),
        "window_cur": len(cur_df),
        "dataset_drift": drift_detected,
        "drift_share": drift_share,
        "report_html": str(html_path),
        "retrain_triggered": retrain_result is not None,
        "mlflow_retrain": retrain_result,
    }
    summary_path = reports_dir / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Drishti Drift] Saved {html_path} and {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drishti Evidently Drift Report & Auto-Retrain Trigger")
    parser.add_argument("--service-id", required=True, help="Service ID to inspect for drift")
    parser.add_argument("--window-ref", type=int, default=500)
    parser.add_argument("--window-cur", type=int, default=200)
    parser.add_argument("--no-auto-retrain", action="store_true", help="Disable auto-retrain on drift")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force auto-retrain trigger even if drift is False (useful for live viva demos)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_report(
        service_id=args.service_id,
        window_ref=args.window_ref,
        window_cur=args.window_cur,
        auto_retrain=not args.no_auto_retrain,
        force_retrain=args.force_retrain,
    )
