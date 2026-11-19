"""Seed NAB dataset series into the Drishti storage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path("nab_dataset/realKnownCause/realKnownCause")
OUTPUT_DIR = Path("data/nab")

AVAILABLE_SERIES: dict[str, str] = {
    "machine_temperature_system_failure": "machine_temperature_system_failure.csv",
    "ambient_temperature_system_failure": "ambient_temperature_system_failure.csv",
    "cpu_utilization_asg_misconfiguration": "cpu_utilization_asg_misconfiguration.csv",
    "nyc_taxi": "nyc_taxi.csv",
    "ec2_request_latency_system_failure": "ec2_request_latency_system_failure.csv",
}


def normalize_series(csv_path: Path) -> pd.DataFrame:
    """Load a NAB CSV and normalize it to ISO timestamps."""

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.rename(columns={"value": "y"}, inplace=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    return df[["timestamp", "y"]]


def seed_series(source_dir: Path, output_dir: Path, selections: list[str]) -> None:
    """Copy and normalize selected series into the data directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in selections:
        if name not in AVAILABLE_SERIES:
            raise ValueError(f"Unknown series '{name}'. Available: {list(AVAILABLE_SERIES)}")
        src = source_dir / AVAILABLE_SERIES[name]
        if not src.exists():
            raise FileNotFoundError(f"Input CSV not found: {src}")
        df = normalize_series(src)
        out_path = output_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Seeded {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed NAB dataset (real known cause series).")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--series", nargs="+", default=list(AVAILABLE_SERIES.keys()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_series(Path(args.source), Path(args.output), args.series)
