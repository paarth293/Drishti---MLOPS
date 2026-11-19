'"""Pydantic models for API payload validation."""'

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TimePoint(BaseModel):
    """Single timestamped measurement for a service."""

    model_config = ConfigDict(extra="forbid")

    ts: datetime
    y: float

    @field_validator("ts")
    @classmethod
    def ensure_timezone(cls, value: datetime) -> datetime:
        """Ensure timestamps are timezone-aware and normalized to UTC."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("y")
    @classmethod
    def ensure_finite(cls, value: float) -> float:
        """Validate that measurements are finite."""
        if not isfinite(value):
            raise ValueError("Measurement value must be finite.")
        return float(value)


class SeriesIngestRequest(BaseModel):
    """Ingestion payload for a single service with time series points."""

    model_config = ConfigDict(extra="forbid")

    service_id: str = Field(min_length=1, max_length=128)
    points: list[TimePoint]

    @model_validator(mode="after")
    def validate_points_not_empty(self) -> SeriesIngestRequest:
        """Ensure each series contains at least one point."""
        if not self.points:
            raise ValueError("Series must contain at least one point.")
        return self


class DetectRequest(BaseModel):
    """Batch detection payload across multiple services."""

    model_config = ConfigDict(extra="forbid")

    series: list[SeriesIngestRequest]
    detector: str | None = None
    alpha: float | None = None


class AnomalyPoint(BaseModel):
    """Detected anomaly enriched with model residual and conformal interval."""

    model_config = ConfigDict(extra="forbid")

    ts: datetime
    y: float
    y_hat: float | None = None
    residual: float | None = None
    q_alpha: float | None = None
    is_anomaly: bool

    @field_validator("ts")
    @classmethod
    def normalize_ts(cls, value: datetime) -> datetime:
        """Normalize timestamps to UTC for storage consistency."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("y", "y_hat", "residual", "q_alpha")
    @classmethod
    def ensure_optional_finite(cls, value: float | None) -> float | None:
        """Validate optional numeric values are finite when provided."""
        if value is None:
            return value
        if not isfinite(value):
            raise ValueError("Numeric values must be finite.")
        return float(value)


class DetectResponse(BaseModel):
    """Detection response bundling anomalies per service."""

    model_config = ConfigDict(extra="forbid")

    service_id: str
    anomalies: list[AnomalyPoint]


class RCAResponse(BaseModel):
    """Ranked root-cause probabilities."""

    model_config = ConfigDict(extra="forbid")

    ranking: list[tuple[str, float]]


class GraphEdge(BaseModel):
    """Weighted directed edge in the service dependency graph."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    weight: float

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        """Ensure edge weights are positive."""
        if value <= 0:
            raise ValueError("Edge weight must be positive.")
        return float(value)


class GraphPayload(BaseModel):
    """Service dependency graph topology description."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[str]
    edges: list[GraphEdge]

    @model_validator(mode="after")
    def ensure_nodes_are_unique(self) -> GraphPayload:
        """Ensure graph nodes are unique to avoid duplication."""
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Graph nodes must be unique.")
        return self
