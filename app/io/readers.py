"""Readers for external datasets and utilities for StatsForecast panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parse_timestamp_column(df: pd.DataFrame) -> pd.Series:
    """Infer and parse a timestamp column from a dataframe.

    Args:
        df: Dataframe loaded from CSV.

    Returns:
        Pandas Series with timezone-aware timestamps.

    Raises:
        ValueError: If no timestamp column can be identified.
    """
    for column in ("timestamp", "ts", "time", "datetime"):
        if column in df.columns:
            return pd.to_datetime(df[column], utc=True)
    if df.index.name in ("timestamp", "ts", "time", "datetime"):
        return pd.to_datetime(df.index, utc=True)
    raise ValueError("CSV must contain a timestamp column named timestamp/ts/time/datetime.")


def _parse_value_column(df: pd.DataFrame) -> pd.Series:
    """Extract numeric values from the dataframe.

    Args:
        df: Dataframe loaded from CSV.

    Returns:
        Numeric value series.

    Raises:
        ValueError: If no numeric column is present or contains invalid data.
    """
    candidate_columns = [col for col in df.columns if col not in {"timestamp", "ts", "time", "datetime"}]
    if not candidate_columns:
        raise ValueError("CSV must contain a numeric value column.")
    series = pd.to_numeric(df[candidate_columns[0]], errors="coerce")
    if series.isna().any():
        raise ValueError("Value column contains non-numeric entries.")
    return series


def load_series_from_csv(path: str | Path, service_id: str) -> pd.DataFrame:
    """Load a time-series CSV and map it to StatsForecast layout.

    Args:
        path: Path to the CSV file.
        service_id: Identifier assigned to the resulting series.

    Returns:
        Dataframe with columns ``unique_id``, ``ds`` and ``y``.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    ds = _parse_timestamp_column(df)
    y = _parse_value_column(df)
    return pd.DataFrame({"unique_id": service_id, "ds": ds, "y": y})


def df_to_panel(series_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert a dictionary of service dataframes into a StatsForecast panel.

    Args:
        series_map: Mapping of service identifiers to their time-series dataframes.

    Returns:
        Concatenated panel dataframe sorted by service id and timestamp.

    Raises:
        ValueError: If any dataframe lacks the required columns.
    """

    frames = []
    for service_id, df in series_map.items():
        if "ds" not in df.columns or "y" not in df.columns:
            raise ValueError(f"DataFrame for {service_id} must contain 'ds' and 'y' columns.")
        frame = df.copy()
        frame["unique_id"] = service_id
        frame = frame[["unique_id", "ds", "y"]]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    panel = pd.concat(frames, ignore_index=True)
    panel.sort_values(["unique_id", "ds"], inplace=True)
    return panel
