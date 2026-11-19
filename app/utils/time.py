"""Time utilities for timezone handling and windows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dateutil import parser


def to_utc(value: datetime) -> datetime:
    """Convert a datetime to UTC timezone.

    Args:
        value: Datetime to convert.

    Returns:
        Timezone-aware UTC datetime.
    """

    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_iso8601(value: str) -> datetime:
    """Parse an ISO8601 datetime string into a timezone-aware datetime.

    Args:
        value: ISO8601 formatted string.

    Returns:
        Parsed UTC datetime.
    """

    dt = parser.isoparse(value)
    return to_utc(dt)


def window_start(dt: datetime, window: timedelta) -> datetime:
    """Return the start of the sliding window ending at dt.

    Args:
        dt: End of the window.
        window: Window duration.

    Returns:
        Start timestamp of the window.
    """

    return to_utc(dt) - window
