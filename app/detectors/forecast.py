"""Forecast-based anomaly detection with conformal calibration."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from app.detectors.baseline import rolling_median_mad_anomaly
from app.detectors.conformal import per_series_calibration

warnings.filterwarnings("ignore", message=".*force_all_finite.*")

try:
    from pmdarima import arima

    PMDARIMA_AVAILABLE = True
except Exception as exc:  # pragma: no cover - executed when dependency missing
    logging.getLogger(__name__).warning(
        "pmdarima_unavailable",
        extra={"reason": str(exc)},
    )
    arima = None  # type: ignore[assignment]
    PMDARIMA_AVAILABLE = False


LOGGER = logging.getLogger(__name__)


def _infer_seasonality(freq: str | None, default: int = 60) -> int:
    """Infer seasonal cycle length from a pandas frequency string.

    Args:
        freq: Pandas frequency alias.
        default: Fallback seasonal length.

    Returns:
        The inferred seasonal cycle length.
    """

    if freq is None:
        return default
    freq = freq.lower()
    if freq.endswith("min") or freq.endswith("t"):
        return 60
    if freq.endswith("h"):
        return 24
    if freq.endswith("s"):
        return 3600
    return default


@dataclass
class ForecastModel:
    """Container for per-series forecasting models."""

    model_type: str
    freq: str | None
    backend: str
    models: dict[str, Any] = field(default_factory=dict)


def fit_forecast(df: pd.DataFrame, model_type: str = "AutoARIMA", freq: str | None = None) -> ForecastModel:
    """Fit AutoARIMA models for each series.

    Args:
        df: Panel dataframe with columns ``unique_id``, ``ds`` et ``y``.
        model_type: Requested model family (currently AutoARIMA).
        freq: Optional inferred sampling frequency.

    Returns:
        Trained forecast model registry.
    """

    backend = "autoarima" if PMDARIMA_AVAILABLE and model_type.lower() == "autoarima" else "holtwinters"
    models: dict[str, Any] = {}
    season_length = _infer_seasonality(freq)
    if backend != "autoarima":
        LOGGER.info("forecast_backend_fallback", extra={"backend": backend})
    for unique_id, group in df.groupby("unique_id"):
        series = group.sort_values("ds")["y"]
        if len(series) < 30:
            LOGGER.info("forecast_fallback_baseline", extra={"service_id": unique_id, "reason": "series too short"})
            continue
        try:
            if backend == "autoarima":  # pragma: no cover - requires optional dependency
                assert PMDARIMA_AVAILABLE and arima is not None  # for type checkers
                model = arima.AutoARIMA(  # pragma: no cover - requires optional dependency
                    start_p=1,
                    start_q=1,
                    max_p=3,
                    max_q=3,
                    seasonal=True,
                    m=season_length,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                ).fit(series)
                models[unique_id] = model
            else:
                seasonal_periods = min(max(2, season_length), max(2, len(series) // 3))
                hw = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                ).fit()
                models[unique_id] = hw
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("forecast_fit_failed", extra={"service_id": unique_id, "reason": str(exc)})
    return ForecastModel(model_type=model_type, freq=freq, backend=backend, models=models)


def predict_one_step(df: pd.DataFrame, model: ForecastModel, freq: str | None = None) -> pd.DataFrame:
    """Generate one-step-ahead predictions for each point in the dataframe.

    Args:
        df: Panel dataframe to forecast.
        model: Fitted forecast model registry.
        freq: Sampling frequency hint.

    Returns:
        Dataframe containing forecasts aligned with the input rows.
    """

    if "unique_id" not in df.columns:
        raise ValueError("Dataframe must contain a 'unique_id' column.")

    predictions: list[pd.DataFrame] = []
    for unique_id, group in df.groupby("unique_id"):
        ordered = group.sort_values("ds").copy()
        fitted_model = model.models.get(unique_id)
        if fitted_model is None:
            # fallback to rolling median baseline predictions
            baseline = rolling_median_mad_anomaly(ordered[["unique_id", "ds", "y"]], window=5, k=3.0)
            ordered["y_hat"] = baseline["y"].rolling(window=5, min_periods=1).mean()
        else:
            if model.backend == "autoarima":  # pragma: no cover - requires optional dependency
                fitted = fitted_model.predict_in_sample()  # pragma: no cover - requires optional dependency
                if len(fitted) != len(ordered):  # pragma: no cover - requires optional dependency
                    fitted = np.concatenate([np.full(len(ordered) - len(fitted), np.nan), fitted])  # pragma: no cover
                ordered["y_hat"] = pd.Series(fitted, index=ordered.index)  # pragma: no cover
            else:
                fitted_series = fitted_model.fittedvalues  # type: ignore[assignment]
                fitted_aligned = pd.Series(fitted_series, index=ordered.index[-len(fitted_series) :])
                ordered["y_hat"] = fitted_aligned.reindex(ordered.index).ffill().bfill()
        ordered["y_hat"] = ordered["y_hat"].bfill().ffill()
        ordered["y_hat"] = ordered["y_hat"].fillna(ordered["y"].mean())
        predictions.append(ordered)
    return pd.concat(predictions, ignore_index=True) if predictions else df.copy()


def detect_with_conformal(
    df: pd.DataFrame,
    model: ForecastModel,
    alpha: float,
    calibration_window: int,
) -> pd.DataFrame:
    """Apply conformal detection using forecast residuals.

    Args:
        df: Panel dataframe to evaluate.
        model: Fitted forecast model registry.
        alpha: Significance level for conformal thresholding.
        calibration_window: Number of recent residuals used for calibration.

    Returns:
        Dataframe with residuals, quantiles, and anomaly labels.
    """

    forecasted = predict_one_step(df, model)
    forecasted["residual"] = forecasted["y"] - forecasted["y_hat"]
    thresholds = per_series_calibration(forecasted, alpha=alpha, window=calibration_window)
    forecasted["q_alpha"] = forecasted["unique_id"].map(thresholds)
    forecasted["is_anomaly"] = forecasted["residual"].abs() > forecasted["q_alpha"]
    return forecasted
