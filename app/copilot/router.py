"""FastAPI router exposing Drishti's GenAI Incident Copilot endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.copilot.explainer import explain_incident

router = APIRouter(tags=["Drishti Copilot"])


class ExplainRequest(BaseModel):
    """Request payload for /rca/explain. All fields are optional for 1-click demo use."""

    k: int = Field(default=5, ge=1, le=20, description="Number of top RCA suspects to analyze.")
    ranking: list[list[Any]] | None = Field(
        default=None,
        description="Optional pre-computed [[service, score], ...] ranking. Auto-fetched if omitted.",
    )
    anomalies: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional sample of recent anomaly objects.",
    )
    metrics_summary: str = Field(default="", description="Optional Prometheus metrics text.")


class ExplainResponse(BaseModel):
    """Structured response from Drishti's LangChain + Gemini incident explainer."""

    project: str
    mode: str
    model: str
    ranking: list[list[Any]]
    explanation: str


@router.post("/rca/explain", response_model=ExplainResponse)
def explain_rca(payload: ExplainRequest | None = None) -> ExplainResponse:
    """Explain the current top-k RCA incident using LangChain + Gemini (or fallback)."""
    req = payload or ExplainRequest()
    ranking = req.ranking
    anomalies = req.anomalies or []

    if ranking is None:
        try:
            from app.api import get_anomalies, rca_topk

            rca_res = rca_topk(k=req.k)
            ranking = [[svc, float(score)] for svc, score in rca_res.ranking]

            if not anomalies and ranking:
                top_svc = str(ranking[0][0])
                pts = get_anomalies(service_id=top_svc, limit=10)
                anomalies = [p.model_dump() if hasattr(p, "model_dump") else dict(p) for p in pts]
        except Exception:  # noqa: BLE001
            ranking = []

    result = explain_incident(
        ranking=ranking or [],
        anomalies=anomalies,
        metrics_summary=req.metrics_summary,
    )
    return ExplainResponse(
        project=result["project"],
        mode=result["mode"],
        model=result["model"],
        ranking=ranking or [],
        explanation=result["explanation"],
    )
