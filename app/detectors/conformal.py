"""Conformal calibration helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class ConformalCalibrator:
    """Quantile-based conformal calibrator."""

    alpha: float
    q_alpha: float = field(init=False, default=0.0)

    def fit(self, residuals: np.ndarray, alpha: float | None = None) -> None:
        """Fit the calibrator on absolute residuals.

        Args:
            residuals: Absolute residual values.
            alpha: Optional override for the significance level.

        Raises:
            ValueError: When the residual array is empty.
        """

        if residuals.size == 0:
            raise ValueError("Residuals array is empty.")
        target_alpha = alpha if alpha is not None else self.alpha
        q_alpha = np.quantile(residuals, 1 - target_alpha, method="higher")
        if q_alpha <= 0:
            q_alpha = float(np.maximum(residuals.max(), 1e-6))
        self.q_alpha = float(q_alpha)
        self.alpha = target_alpha

    def predict_threshold(self) -> float:
        """Return the calibrated quantile threshold.

        Returns:
            Calibrated quantile threshold.
        """

        return self.q_alpha


def compute_residuals(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """Compute absolute residuals.

    Args:
        y: Observed values.
        y_hat: Predicted values.

    Returns:
        Absolute residual values.

    Raises:
        ValueError: If the arrays have different shapes.
    """

    if y.shape != y_hat.shape:
        raise ValueError("y and y_hat must have the same shape.")
    return np.abs(y - y_hat)


def per_series_calibration(panel_df: pd.DataFrame, alpha: float, window: int) -> dict[str, float]:
    """Calibrate conformal thresholds per series.

    Args:
        panel_df: Dataframe containing ``unique_id`` and ``residual`` columns.
        alpha: Significance level.
        window: Calibration window size.

    Returns:
        Mapping from service id to conformal threshold.
    """

    thresholds: dict[str, float] = {}
    global_residuals = panel_df["residual"].abs().dropna()
    if global_residuals.empty:
        global_default = 1.0
    else:
        global_default = float(np.quantile(global_residuals.tail(window), 1 - alpha, method="higher"))
    coverage_records = []

    for unique_id, group in panel_df.groupby("unique_id"):
        residuals = group["residual"].abs().dropna()
        if residuals.empty:
            thresholds[unique_id] = global_default
            continue
        windowed = residuals.tail(window)
        calibrator = ConformalCalibrator(alpha=alpha)
        calibrator.fit(windowed.to_numpy())
        thresholds[unique_id] = calibrator.predict_threshold()
        coverage = float((windowed <= thresholds[unique_id]).mean())
        coverage_records.append((unique_id, coverage))

    if coverage_records:
        average_coverage = np.mean([item[1] for item in coverage_records])
        LOGGER.info(
            "conformal_coverage",
            extra={
                "average_coverage": average_coverage,
                "target": 1 - alpha,
                "per_series": coverage_records,
            },
        )
    return thresholds
