"""Baseline anomaly detectors based on robust statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_median_mad_anomaly(df: pd.DataFrame, window: int, k: float) -> pd.DataFrame:
    """Detect anomalies using rolling median and MAD.

    Args:
        df: Panel dataframe with ``unique_id``, ``ds`` et ``y``.
        window: Rolling window size.
        k: Threshold multiplier for MAD scores.

    Returns:
        Dataframe with anomaly annotations ``score``, ``q_alpha`` et ``is_anomaly``.

    Raises:
        ValueError: When required columns are missing.
    """

    if "unique_id" not in df.columns or "y" not in df.columns or "ds" not in df.columns:
        raise ValueError("Dataframe must contain 'unique_id', 'ds', and 'y'.")

    outputs = []
    for _unique_id, group in df.groupby("unique_id", sort=False):
        group = group.sort_values("ds").copy()
        rolling_median = group["y"].rolling(window=window, min_periods=max(1, window // 2)).median()
        deviations = (group["y"] - rolling_median).abs()
        mad = deviations.rolling(window=window, min_periods=max(1, window // 2)).median()
        mad_scaled = 1.4826 * mad.replace(0.0, np.nan)
        score = deviations / mad_scaled
        group["y_hat"] = rolling_median
        group["residual"] = group["y"] - group["y_hat"]
        group["score"] = score.fillna(0.0)
        group["q_alpha"] = (k * mad_scaled).fillna(k)
        group["is_anomaly"] = group["score"] > k
        outputs.append(group)

    return pd.concat(outputs, ignore_index=True) if outputs else df.copy()


def zscore_rolling(df: pd.DataFrame, window: int, threshold: float) -> pd.DataFrame:
    """Detect anomalies using rolling mean and standard deviation.

    Args:
        df: Panel dataframe with ``unique_id``, ``ds`` et ``y``.
        window: Rolling window size.
        threshold: Z-score threshold above which anomalies are flagged.

    Returns:
        Dataframe with z-score ``score`` and ``is_anomaly`` indicator.

    Raises:
        ValueError: When required columns are missing.
    """

    if "unique_id" not in df.columns or "y" not in df.columns or "ds" not in df.columns:
        raise ValueError("Dataframe must contain 'unique_id', 'ds', and 'y'.")

    outputs = []
    for _unique_id, group in df.groupby("unique_id", sort=False):
        group = group.sort_values("ds").copy()
        rolling_mean = group["y"].rolling(window=window, min_periods=max(1, window // 2)).mean()
        rolling_std = group["y"].rolling(window=window, min_periods=max(1, window // 2)).std(ddof=0)
        rolling_std = rolling_std.replace(0.0, np.nan)
        z_scores = (group["y"] - rolling_mean) / rolling_std
        group["y_hat"] = rolling_mean
        group["residual"] = group["y"] - group["y_hat"]
        group["score"] = z_scores.fillna(0.0).abs()
        group["q_alpha"] = (threshold * rolling_std).abs().fillna(threshold)
        group["is_anomaly"] = group["score"] > threshold
        outputs.append(group)

    return pd.concat(outputs, ignore_index=True) if outputs else df.copy()
