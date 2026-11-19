"""Unit tests for Drishti MLflow tracking features and data loader."""

import pandas as pd

from scripts.mlflow_tracking import build_features, load_training_dataframe


def test_build_features():
    df = pd.DataFrame({"y": [10.0, 20.0, 30.0, 40.0, 50.0]})
    feat = build_features(df)
    assert "y" in feat.columns
    assert "rolling_mean_10" in feat.columns
    assert "rolling_std_10" in feat.columns
    assert "diff_1" in feat.columns
    assert len(feat) == 5


def test_load_training_dataframe_synthetic():
    df = load_training_dataframe(service_id="test_service")
    assert not df.empty
    assert "service_id" in df.columns
    assert "ts" in df.columns
    assert "y" in df.columns
