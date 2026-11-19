"""Tests for time utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.utils.time import parse_iso8601, to_utc, window_start


def test_to_utc_handles_naive_datetime() -> None:
    naive = datetime(2025, 1, 1, 12, 0, 0)
    converted = to_utc(naive)
    assert converted.tzinfo == UTC
    assert converted.hour == 12


def test_to_utc_handles_aware_datetime() -> None:
    aware = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    converted = to_utc(aware)
    assert converted.tzinfo == UTC


@pytest.mark.parametrize(
    ("value", "expected_hour"),
    [
        ("2025-01-01T12:00:00Z", 12),
        ("2025-01-01T12:00:00+02:00", 10),
    ],
)
def test_parse_iso8601_normalises_to_utc(value: str, expected_hour: int) -> None:
    parsed = parse_iso8601(value)
    assert parsed.tzinfo == UTC
    assert parsed.hour == expected_hour


def test_window_start_returns_correct_delta() -> None:
    end = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    start = window_start(end, timedelta(hours=2))
    assert start == datetime(2025, 1, 1, 10, 0, tzinfo=UTC)
