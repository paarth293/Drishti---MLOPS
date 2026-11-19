"""Logging configuration utilities for JSON-structured logs."""

from __future__ import annotations

import json
import logging
from logging.config import dictConfig
from typing import Any


class JsonFormatter(logging.Formatter):
    """Serialize log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record into JSON.

        Args:
            record: Log record emitted by the logging subsystem.

        Returns:
            A JSON string containing the structured log payload.
        """

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging with JSON formatting.

    Args:
        level: Minimum log level applied to the root logger.
    """

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JsonFormatter,
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "level": level,
                }
            },
            "loggers": {
                "": {"handlers": ["default"], "level": level},
                "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": level, "propagate": False},
            },
        }
    )
