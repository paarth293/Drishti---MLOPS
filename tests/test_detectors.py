"""Tests for baseline and forecast detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.detectors.baseline import rolling_median_mad_anomaly, zscore_rolling
from app.detectors.forecast import (
    PMDARIMA_AVAILABLE,
    _infer_seasonality,
    detect_with_conformal,
    fit_forecast,
    predict_one_step,
)

DEFAULT_ANOMALY_INDEX = 90


def _build_panel(periods: int = 120, anomaly_idx: int = DEFAULT_ANOMALY_INDEX) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=periods, freq="min")
    rng = np.random.default_rng(seed=42)
    values = 20 + np.sin(np.linspace(0, 12, periods)) + rng.normal(0, 0.2, periods)
    values[anomaly_idx] += 10  # inject spike
    df = pd.DataFrame({"unique_id": "service_a", "ds": timestamps, "y": values})
    return df


def test_rolling_median_detects_anomaly() -> None:
    panel = _build_panel()
    result = rolling_median_mad_anomaly(panel, window=20, k=3.0)
    anomalies = result[result["is_anomaly"]]
    assert not anomalies.empty
    injected_ts = panel["ds"].iloc[DEFAULT_ANOMALY_INDEX]
    assert injected_ts in anomalies["ds"].to_list()


def test_forecast_detector_flags_spike() -> None:
    panel = _build_panel(periods=150, anomaly_idx=120)
    model = fit_forecast(panel, model_type="AutoARIMA", freq="min")
    detected = detect_with_conformal(panel, model=model, alpha=0.1, calibration_window=50)
    anomalies = detected[detected["is_anomaly"]]
    assert not anomalies.empty
    injected_ts = panel["ds"].iloc[120]
    assert injected_ts in anomalies["ds"].to_list()


def test_zscore_rolling_detects_shift() -> None:
    panel = _build_panel()
    result = zscore_rolling(panel, window=15, threshold=2.5)
    injected_ts = panel["ds"].iloc[DEFAULT_ANOMALY_INDEX]
    flagged = result[result["is_anomaly"]]
    assert injected_ts in flagged["ds"].to_list()


def test_predict_one_step_handles_short_series() -> None:
    panel = _build_panel(periods=40, anomaly_idx=10)
    model = fit_forecast(panel, model_type="AutoARIMA", freq="min")
    assert model.backend in {"holtwinters", "autoarima"}
    predicted = predict_one_step(panel, model)
    assert "y_hat" in predicted.columns
    assert not predicted["y_hat"].isna().all()


def test_fit_forecast_skips_short_series() -> None:
    panel = _build_panel(periods=20, anomaly_idx=5)
    model = fit_forecast(panel, model_type="AutoARIMA", freq="min")
    assert model.models == {}


def test_infer_seasonality_variants() -> None:
    assert _infer_seasonality(None) == 60
    assert _infer_seasonality("5min") == 60
    assert _infer_seasonality("2H") == 24
    assert _infer_seasonality("30S") == 3600


@pytest.mark.skipif(not PMDARIMA_AVAILABLE, reason="pmdarima backend unavailable")
def test_fit_forecast_autoarima_backend() -> None:
    panel = _build_panel(periods=180, anomaly_idx=150)
    model = fit_forecast(panel, model_type="AutoARIMA", freq="min")
    assert model.backend == "autoarima"
    predicted = predict_one_step(panel, model)
    assert predicted["y_hat"].isna().sum() < 5


def test_fit_forecast_holtwinters_backend() -> None:
    panel = _build_panel(periods=90, anomaly_idx=60)
    model = fit_forecast(panel, model_type="mstl", freq="min")
    assert model.backend == "holtwinters"
    predicted = predict_one_step(panel, model)
    assert not predicted["y_hat"].isna().any()
