"""Tests for conformal calibration utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.detectors.conformal import ConformalCalibrator, compute_residuals, per_series_calibration


def test_conformal_calibrator_threshold() -> None:
    residuals = np.abs(np.random.default_rng(0).normal(0, 1, 1000))
    calibrator = ConformalCalibrator(alpha=0.1)
    calibrator.fit(residuals)
    threshold = calibrator.predict_threshold()
    assert threshold > 0
    coverage = (residuals <= threshold).mean()
    assert 0.85 <= coverage <= 0.95


def test_per_series_calibration_matches_target() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for service_id in ("svc_a", "svc_b"):
        residuals = rng.normal(0, 1, 600)
        timestamps = pd.date_range("2025-01-01", periods=600, freq="min")
        for ts, resid in zip(timestamps, residuals, strict=False):
            rows.append({"unique_id": service_id, "ds": ts, "residual": resid})
    panel = pd.DataFrame(rows)
    alpha = 0.1
    thresholds = per_series_calibration(panel, alpha=alpha, window=500)
    assert set(thresholds) == {"svc_a", "svc_b"}
    for service_id, threshold in thresholds.items():
        residuals = panel[panel["unique_id"] == service_id]["residual"].tail(500)
        coverage = (residuals.abs() <= threshold).mean()
        assert abs(coverage - (1 - alpha)) < 0.05


def test_compute_residuals_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_residuals(np.array([1.0, 2.0]), np.array([1.0]))


def test_conformal_calibrator_empty_residuals() -> None:
    calibrator = ConformalCalibrator(alpha=0.1)
    with pytest.raises(ValueError):
        calibrator.fit(np.array([]))
