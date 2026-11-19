"""Drishti MLflow Experiment Tracking & @champion Model Registry Pipeline.

Mirrors the MLOps workflow:
1. Loads time-series measurements from Drishti's SQLite/Postgres DB (with synthetic fallback).
2. Builds a leakage-free scikit-learn Pipeline (StandardScaler + IsolationForest).
3. Logs parameters, metrics, and Drishti artifacts (nab_summary.json, latest.html) to MLflow.
4. Registers `drishti-anomaly-detector` in the MLflow Model Registry and promotes `@champion`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

load_dotenv()

MODEL_NAME = "drishti-anomaly-detector"
EXPERIMENT_NAME = "drishti-aiops-monitoring"


def _resolve_db_url() -> str:
    """Resolve database URL from app.settings or environment."""
    try:
        from app.settings import load_settings

        return load_settings().effective_db_url
    except Exception:  # noqa: BLE001
        return os.getenv("DB_URL", "sqlite:///./data/aiops.db")


def load_training_dataframe(service_id: str | None = None, limit: int = 2000) -> pd.DataFrame:
    """Load measurements from SQLite/Postgres, or create a small synthetic series if empty."""
    db_url = _resolve_db_url()
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            if service_id:
                query = text(
                    "SELECT service_id, ts, y FROM measurements "
                    "WHERE service_id = :sid ORDER BY ts DESC LIMIT :lim"
                )
                df = pd.read_sql(query, conn, params={"sid": service_id, "lim": limit})
            else:
                query = text(
                    "SELECT service_id, ts, y FROM measurements ORDER BY ts DESC LIMIT :lim"
                )
                df = pd.read_sql(query, conn, params={"lim": limit})
        if not df.empty and len(df) >= 20:
            return df.sort_values("ts").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[Drishti MLflow] DB read skipped ({exc}); using synthetic series.")

    rng = np.random.default_rng(42)
    normal = rng.normal(loc=50.0, scale=4.0, size=300)
    spikes = np.array([92.0, 98.5, 5.2, 105.0, 88.3])
    values = np.concatenate([normal, spikes])
    return pd.DataFrame(
        {
            "service_id": service_id or "synthetic_checkout",
            "ts": pd.date_range("2026-01-01", periods=len(values), freq="min"),
            "y": values,
        }
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple rolling features for time-series anomaly detection."""
    feat = pd.DataFrame(index=df.index)
    feat["y"] = df["y"].astype(float)
    feat["rolling_mean_10"] = feat["y"].rolling(window=10, min_periods=1).mean()
    feat["rolling_std_10"] = feat["y"].rolling(window=10, min_periods=1).std().fillna(0.0)
    feat["diff_1"] = feat["y"].diff().fillna(0.0)
    return feat


def run_mlflow_pipeline(
    service_id: str | None = None,
    contamination: float = 0.05,
    n_estimators: int = 100,
    promote_champion: bool = True,
    trigger_reason: str = "manual_run",
) -> dict[str, Any]:
    """Train IsolationForest pipeline, log to MLflow, and optionally promote @champion."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_training_dataframe(service_id=service_id)
    features = build_features(df)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(features)
    preds = pipeline.predict(features)  # -1 = anomaly, 1 = normal
    scores = pipeline.decision_function(features)

    anomaly_count = int((preds == -1).sum())
    anomaly_rate = float(anomaly_count / max(len(preds), 1))

    # Load repo model.yaml parameters if available
    model_yaml = Path("configs/model.yaml")
    yaml_cfg: dict[str, Any] = {}
    if model_yaml.exists():
        yaml_cfg = yaml.safe_load(model_yaml.read_text(encoding="utf-8")) or {}

    with mlflow.start_run(run_name=f"drishti-{trigger_reason}") as run:
        mlflow.log_params(
            {
                "project": "Drishti",
                "trigger_reason": trigger_reason,
                "service_id": service_id or "all_services",
                "model_type": "IsolationForest",
                "contamination": contamination,
                "n_estimators": n_estimators,
                "training_rows": len(features),
            }
        )
        if yaml_cfg:
            mlflow.log_param("config_keys", ",".join(sorted(yaml_cfg.keys())))

        mlflow.log_metrics(
            {
                "detected_anomalies": float(anomaly_count),
                "anomaly_rate": anomaly_rate,
                "mean_decision_score": float(np.mean(scores)),
                "min_decision_score": float(np.min(scores)),
            }
        )

        # Attach existing showcase/drift artifacts if present
        for artifact_path in [
            Path("reports/nab_summary.json"),
            Path("reports/latest.html"),
            Path("reports/drift_summary.json"),
        ]:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="drishti_reports")

        # Also log NAB summary metrics if available
        nab_summary = Path("reports/nab_summary.json")
        if nab_summary.exists():
            try:
                summary_data = json.loads(nab_summary.read_text(encoding="utf-8"))
                rca_list = summary_data.get("rca_topk", [])
                if rca_list:
                    mlflow.log_metric("top1_rca_score", float(rca_list[0][1]))
            except Exception:  # noqa: BLE001
                pass

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    champion_version = None
    if promote_champion:
        client = MlflowClient(tracking_uri=tracking_uri)
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            champion_version = latest.version
            client.set_registered_model_alias(MODEL_NAME, "champion", champion_version)

    result = {
        "tracking_uri": tracking_uri,
        "experiment": EXPERIMENT_NAME,
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "champion_version": champion_version,
        "training_rows": len(features),
        "detected_anomalies": anomaly_count,
        "anomaly_rate": round(anomaly_rate, 4),
    }
    print("[Drishti MLflow] Run complete:", json.dumps(result, indent=2))
    return result


def load_champion_model() -> Any:
    """Load the promoted @champion model from MLflow Model Registry."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drishti MLflow Tracking & Registry Runner")
    parser.add_argument("--service-id", default=None, help="Optional service_id to train on")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--no-promote", action="store_true", help="Skip @champion alias promotion")
    parser.add_argument("--trigger-reason", default="manual_run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mlflow_pipeline(
        service_id=args.service_id,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        promote_champion=not args.no_promote,
        trigger_reason=args.trigger_reason,
    )
