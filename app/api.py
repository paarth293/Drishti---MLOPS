"""FastAPI routes for Drishti MLOps Platform."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import PlainTextResponse

from app.detectors.baseline import rolling_median_mad_anomaly
from app.detectors.forecast import detect_with_conformal, fit_forecast
from app.io.otel_tempo import TempoClient
from app.io.prometheus import PrometheusClient
from app.io.writers import (
    count_anomalies,
    count_measurements,
    init_db,
    list_services_with_anomalies,
    query_anomalies,
    query_measurements,
    write_anomalies,
    write_measurements,
)
from app.models.schemas import (
    AnomalyPoint,
    DetectRequest,
    DetectResponse,
    GraphPayload,
    RCAResponse,
    SeriesIngestRequest,
    TimePoint,
)
from app.rca.graph_builder import build_graph_from_config, merge_graphs, normalize_edge_weights
from app.rca.rca_ranker import rca_score
from app.settings import Settings, load_settings
from app.utils.time import parse_iso8601, to_utc

LOGGER = logging.getLogger(__name__)


def _infer_frequency(points: list[TimePoint]) -> str | None:
    """Infer sampling frequency from consecutive points.

    Args:
        points: Time-series points ordered arbitrarily.

    Returns:
        Pandas-compatible frequency string when enough points are available.
    """

    if len(points) < 2:
        return None
    deltas = []
    sorted_points = sorted(points, key=lambda item: item.ts)
    for first, second in zip(sorted_points, sorted_points[1:], strict=False):
        deltas.append((second.ts - first.ts).total_seconds())
    if not deltas:
        return None
    median_delta = float(np.median(deltas))
    if median_delta >= 3600:
        return f"{int(median_delta // 3600)}H"
    if median_delta >= 60:
        return f"{int(median_delta // 60)}T"
    if median_delta > 0:
        return f"{int(median_delta)}S"
    return None


def _points_to_dataframe(service: SeriesIngestRequest) -> pd.DataFrame:
    """Convert a series request to a StatsForecast-compatible dataframe.

    Args:
        service: Ingestion payload for a given service.

    Returns:
        Dataframe containing ``unique_id``, ``ds`` et ``y``.
    """

    rows = [{"unique_id": service.service_id, "ds": point.ts, "y": point.y} for point in service.points]
    return pd.DataFrame(rows)


@dataclass
class ApiState:
    """Mutable API state shared across requests."""

    settings: Settings
    base_graph: nx.DiGraph
    graph: nx.DiGraph
    local_scores: dict[str, tuple[float, datetime]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    tempo_client: TempoClient | None = None
    prometheus_client: PrometheusClient | None = None

    def refresh_graph_from_traces(self) -> None:
        """Refresh dependency graph by merging Tempo traces when enabled."""

        if not self.tempo_client or not self.tempo_client.enabled:
            return
        traces = self.tempo_client.fetch_recent_traces(self.settings.rca_window_minutes)
        trace_graph = self.tempo_client.build_service_graph_from_traces(traces)
        merged = merge_graphs(self.base_graph, trace_graph)
        self.graph = normalize_edge_weights(merged)


def create_state() -> ApiState:
    """Create API state with configuration and integrations.

    Returns:
        Initialised API state carrying configuration and graph data.
    """

    settings = load_settings()
    init_db(settings)
    graph_config = settings.data_config_map.get("graph", {})
    base_graph = normalize_edge_weights(build_graph_from_config(graph_config))
    tempo_client = None
    if settings.enable_otel and settings.tempo_base_url:
        tempo_client = TempoClient(base_url=settings.tempo_base_url, enabled=True)
    prom_client = None
    if settings.enable_prometheus and settings.prometheus_base_url:
        prom_client = PrometheusClient(base_url=str(settings.prometheus_base_url))
    api_state = ApiState(
        settings=settings,
        base_graph=base_graph,
        graph=base_graph,
        tempo_client=tempo_client,
        prometheus_client=prom_client,
    )
    _hydrate_state_from_storage(api_state)
    return api_state


def _hydrate_state_from_storage(api_state: ApiState) -> None:
    """Populate in-memory state from persisted anomalies/measurements."""

    try:
        api_state.metrics["ingested_points"] = float(count_measurements())
        api_state.metrics["anomaly_count"] = float(count_anomalies())
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("hydrate_metrics_failed", extra={"error": str(exc)})

    try:
        service_ids = list_services_with_anomalies()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("hydrate_local_scores_failed", extra={"error": str(exc)})
        return

    now = datetime.now(UTC)
    for service_id in service_ids:
        anomalies = query_anomalies(service_id=service_id, since=None, limit=200)
        ratios: list[float] = []
        for anomaly in anomalies:
            if anomaly.q_alpha and anomaly.q_alpha > 0 and anomaly.residual is not None:
                ratios.append(min(abs(anomaly.residual) / anomaly.q_alpha, 3.0))
        if ratios:
            api_state.local_scores[service_id] = (float(np.mean(ratios)), now)


state = create_state()
router = APIRouter()


async def get_state() -> ApiState:
    """Dependency returning global state."""

    return state


@router.get("/health")
async def healthcheck() -> dict:
    """Return service health indicator.

    Returns:
        JSON payload with status key.
    """

    return {"status": "ok"}


@router.post("/ingest/series")
async def ingest_series(payload: SeriesIngestRequest, api_state: ApiState = Depends(get_state)) -> dict:
    """Ingest a batch of time-series points.

    Args:
        payload: Ingestion request containing service data.
        api_state: Shared API state dependency.

    Returns:
        JSON payload containing number of ingested points.

    Raises:
        HTTPException: If payload exceeds size limits.
    """

    settings = api_state.settings
    if len(payload.points) > settings.ingest_max_points:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Too many points in payload.")
    ingested = write_measurements(payload.service_id, payload.points)
    api_state.metrics["ingested_points"] += ingested
    LOGGER.info("ingest_series", extra={"service_id": payload.service_id, "points": len(payload.points)})
    return {"ingested": ingested}


@router.post("/detect/batch", response_model=list[DetectResponse])
async def detect_batch(payload: DetectRequest, api_state: ApiState = Depends(get_state)) -> list[DetectResponse]:
    """Detect anomalies for multiple services.

    Args:
        payload: Batch detection request.
        api_state: Shared API state dependency.

    Returns:
        List of detection responses per service.

    Raises:
        HTTPException: If inputs are invalid or size limits exceeded.
    """

    if not payload.series:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No series provided.")
    settings = api_state.settings

    start_time = time.perf_counter()
    panel_frames = []
    freq_candidates = []
    for series in payload.series:
        if len(series.points) > settings.ingest_max_points:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Too many points for service {series.service_id}",
            )
        write_measurements(series.service_id, series.points)
        panel_frames.append(_points_to_dataframe(series))
        freq = _infer_frequency(series.points)
        if freq:
            freq_candidates.append(freq)

    if not panel_frames:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty panel data.")

    panel_df = pd.concat(panel_frames, ignore_index=True)
    detector = (payload.detector or settings.model_config_map.get("detector", {}).get("type") or "forecast").lower()
    alpha = payload.alpha or settings.conformal_alpha

    if detector == "baseline":
        baseline_cfg = settings.model_config_map.get("baseline", {})
        window = int(baseline_cfg.get("rolling_window", 120))
        k = float(baseline_cfg.get("mad_k", 4.0))
        detection_df = rolling_median_mad_anomaly(panel_df, window=window, k=k)
        detection_df["y_hat"] = detection_df["y_hat"].fillna(detection_df["y"])
        detection_df["residual"] = detection_df["residual"].fillna(detection_df["y"] - detection_df["y_hat"])
        detection_df["q_alpha"] = detection_df["q_alpha"].fillna(k)
        detection_df["is_anomaly"] = detection_df["is_anomaly"].fillna(False)
    else:
        freq = freq_candidates[0] if freq_candidates else None
        model = fit_forecast(panel_df, model_type="AutoARIMA", freq=freq)
        conformal_cfg = settings.model_config_map.get("conformal", {})
        calibration_window = int(conformal_cfg.get("calibration_window", 500))
        detection_df = detect_with_conformal(panel_df, model=model, alpha=alpha, calibration_window=calibration_window)

    responses: list[DetectResponse] = []
    recorded_ts = datetime.now(UTC)
    for service_id, group in detection_df.groupby("unique_id"):
        group = group.sort_values("ds")
        anomalies = []
        ratios = []
        for _, row in group.iterrows():
            anomaly = AnomalyPoint(
                ts=to_utc(row["ds"]),
                y=float(row["y"]),
                y_hat=float(row["y_hat"]) if not pd.isna(row.get("y_hat")) else None,
                residual=float(row["residual"]) if not pd.isna(row.get("residual")) else None,
                q_alpha=float(row["q_alpha"]) if not pd.isna(row.get("q_alpha")) else None,
                is_anomaly=bool(row.get("is_anomaly", False)),
            )
            if anomaly.q_alpha and anomaly.q_alpha > 0 and anomaly.residual is not None:
                ratios.append(min(abs(anomaly.residual) / anomaly.q_alpha, 3.0))
            if anomaly.is_anomaly:
                anomalies.append(anomaly)
        if anomalies:
            write_anomalies(service_id, anomalies)
        mean_ratio = float(np.mean(ratios)) if ratios else 0.0
        api_state.local_scores[service_id] = (mean_ratio, recorded_ts)
        api_state.metrics["anomaly_count"] += len(anomalies)
        responses.append(DetectResponse(service_id=service_id, anomalies=anomalies))

    elapsed = time.perf_counter() - start_time
    api_state.metrics["last_detection_latency_sec"] = elapsed
    LOGGER.info(
        "detect_batch_completed",
        extra={
            "series": len(payload.series),
            "elapsed_sec": elapsed,
            "detector": detector,
            "alpha": alpha,
        },
    )
    return responses


@router.get("/anomalies", response_model=list[AnomalyPoint])
async def get_anomalies(
    service_id: str = Query(..., description="Service identifier"),
    since: str | None = Query(None, description="ISO8601 lower bound"),
    limit: int = Query(100, ge=1, le=1000),
    api_state: ApiState = Depends(get_state),
) -> list[AnomalyPoint]:
    """Return persisted anomalies.

    Args:
        service_id: Service identifier.
        since: Optional ISO8601 lower bound.
        limit: Maximum number of items to return.
        api_state: Shared API state dependency.

    Returns:
        List of anomaly points.
    """

    since_dt = parse_iso8601(since) if since else None
    anomalies = query_anomalies(service_id=service_id, since=since_dt, limit=limit)
    LOGGER.info("anomalies_fetch", extra={"service_id": service_id, "count": len(anomalies)})
    return anomalies


@router.get("/graph", response_model=GraphPayload)
async def get_graph(api_state: ApiState = Depends(get_state)) -> GraphPayload:
    """Return the current dependency graph.

    Args:
        api_state: Shared API state dependency.

    Returns:
        Graph payload with nodes and edges.
    """

    api_state.refresh_graph_from_traces()
    graph = api_state.graph
    nodes = list(graph.nodes)
    edges = [
        {"source": source, "target": target, "weight": graph[source][target].get("weight", 1.0)}
        for source, target in graph.edges
    ]
    return GraphPayload(nodes=nodes, edges=edges)


@router.get("/measurements", response_model=list[TimePoint])
async def get_measurements(
    service_id: str = Query(..., description="Service identifier"),
    since: str | None = Query(None, description="ISO8601 lower bound"),
    limit: int = Query(1000, ge=1, le=5000),
) -> list[TimePoint]:
    """Return persisted measurements."""

    since_dt = parse_iso8601(since) if since else None
    return query_measurements(service_id=service_id, since=since_dt, limit=limit)


@router.post("/graph", response_model=GraphPayload)
async def update_graph(payload: GraphPayload, api_state: ApiState = Depends(get_state)) -> GraphPayload:
    """Merge the provided graph with the current configuration.

    Args:
        payload: Graph payload to merge.
        api_state: Shared API state dependency.

    Returns:
        Updated graph payload.
    """

    incoming = build_graph_from_config({"nodes": payload.nodes, "edges": [edge.model_dump() for edge in payload.edges]})
    merged = merge_graphs(api_state.graph, incoming)
    api_state.graph = normalize_edge_weights(merged)
    LOGGER.info("graph_updated", extra={"nodes": len(api_state.graph.nodes), "edges": len(api_state.graph.edges)})
    graph = api_state.graph
    nodes = list(graph.nodes)
    edges = [
        {"source": source, "target": target, "weight": graph[source][target].get("weight", 1.0)}
        for source, target in graph.edges
    ]
    return GraphPayload(nodes=nodes, edges=edges)


@router.get("/rca/topk", response_model=RCAResponse)
async def rca_topk(
    k: int = Query(5, ge=1, le=20),
    api_state: ApiState = Depends(get_state),
) -> RCAResponse:
    """Return the top-k probable root causes.

    Args:
        k: Number of ranked services to return.
        api_state: Shared API state dependency.

    Returns:
        RCA response containing the top-k ranking.
    """

    now = datetime.now(UTC)
    alpha = api_state.settings.rca_alpha
    half_life = timedelta(minutes=api_state.settings.rca_half_life_min)
    ranking = rca_score(api_state.graph, api_state.local_scores, now, alpha=alpha, half_life=half_life)
    return RCAResponse(ranking=ranking[:k])


@router.get("/metrics")
async def metrics(api_state: ApiState = Depends(get_state)) -> Response:
    """Expose Prometheus-style metrics.

    Args:
        api_state: Shared API state dependency.

    Returns:
        Plain text response containing Prometheus metrics.
    """

    lines = [
        "# HELP drishti_ingested_points Total ingested points.",
        "# TYPE drishti_ingested_points counter",
        f"drishti_ingested_points {api_state.metrics.get('ingested_points', 0)}",
        "# HELP drishti_anomaly_count Detected anomalies.",
        "# TYPE drishti_anomaly_count counter",
        f"drishti_anomaly_count {api_state.metrics.get('anomaly_count', 0)}",
        "# HELP drishti_last_detection_latency_seconds Last detection latency in seconds.",
        "# TYPE drishti_last_detection_latency_seconds gauge",
        f"drishti_last_detection_latency_seconds {api_state.metrics.get('last_detection_latency_sec', 0.0)}",
        f"aiops_copilot_anomaly_count {api_state.metrics.get('anomaly_count', 0)}",
    ]
    return PlainTextResponse("\n".join(lines))
