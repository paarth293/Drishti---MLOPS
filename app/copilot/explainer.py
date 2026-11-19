"""Drishti GenAI Incident Explainer using LangChain LCEL + Google Gemini.

Mirrors the LCEL pipeline pattern (`prompt | model | output_parser`) using
`ChatPromptTemplate`, `ChatGoogleGenerativeAI`, and `StrOutputParser`, with
an automatic deterministic fallback when `GEMINI_API_KEY` is not configured.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _fallback_explanation(
    ranking: list[tuple[str, float]] | list[list[Any]],
    anomalies: list[dict[str, Any]],
    metrics_summary: str = "",
) -> str:
    """Return a structured, deterministic explanation when Gemini API is unavailable."""
    if not ranking:
        return (
            "[Drishti Fallback Mode — set GEMINI_API_KEY in .env for live Gemini output]\n\n"
            "No active root-cause candidates were found in the RCA ranking. "
            "Run anomaly detection (`POST /detect/batch` or `make nab-showcase`) first."
        )

    top_service, top_score = ranking[0][0], float(ranking[0][1])
    runners_up = ", ".join(f"{svc} ({float(score):.3f})" for svc, score in ranking[1:4]) or "None"
    anomaly_count = len(anomalies)

    lines = [
        "[Drishti Fallback Mode — set GEMINI_API_KEY in .env for live Gemini output]",
        "",
        f"1. **Primary Root-Cause Suspect:** `{top_service}` (RCA score: {top_score:.3f}).",
        f"2. **Secondary Correlated Services:** {runners_up}.",
        f"3. **Observed Anomaly Context:** {anomaly_count} recent anomaly point(s) inspected.",
        "4. **Recommended Next Steps:**",
        f"   - Inspect recent latency, error logs, and CPU/memory spikes on `{top_service}`.",
        "   - Run `python scripts/run_evidently_report.py --service-id "
        f"{top_service}` to check for input data drift.",
        "   - Verify upstream/downstream edges in the Drishti Dependency Graph tab.",
    ]
    if metrics_summary:
        lines.append(f"5. **Prometheus Metrics Snapshot:** {metrics_summary[:240]}")
    return "\n".join(lines)


def explain_incident(
    ranking: list[tuple[str, float]] | list[list[Any]],
    anomalies: list[dict[str, Any]] | None = None,
    metrics_summary: str = "",
) -> dict[str, Any]:
    """Generate a plain-English root-cause analysis and remediation plan."""
    anomalies = anomalies or []
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    if not api_key or api_key == "your_gemini_api_key_here":
        return {
            "project": "Drishti",
            "mode": "fallback",
            "model": "deterministic-fallback",
            "explanation": _fallback_explanation(ranking, anomalies, metrics_summary),
        }

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Drishti, an expert AIOps & MLOps Incident Copilot. "
                    "Given microservices Root Cause Analysis (RCA) rankings and recent anomalies, "
                    "produce a concise, structured incident brief with: "
                    "(1) Likely Root Cause, (2) Blast Radius / Affected Services, "
                    "(3) Immediate Remediation Steps, and (4) MLOps Monitoring Check.",
                ),
                (
                    "human",
                    "Top-K RCA Ranking (service, score):\n{ranking}\n\n"
                    "Recent Anomalies Sample:\n{anomalies}\n\n"
                    "Metrics Snapshot:\n{metrics_summary}",
                ),
            ]
        )

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
        )
        parser = StrOutputParser()

        # LCEL pipeline: prompt | model | output_parser
        chain = prompt | llm | parser
        explanation = chain.invoke(
            {
                "ranking": str(ranking[:10]),
                "anomalies": str(anomalies[:10]),
                "metrics_summary": metrics_summary or "Not provided",
            }
        )
        return {
            "project": "Drishti",
            "mode": "gemini",
            "model": model_name,
            "explanation": explanation,
        }
    except Exception as exc:  # noqa: BLE001
        fallback = _fallback_explanation(ranking, anomalies, metrics_summary)
        return {
            "project": "Drishti",
            "mode": "fallback",
            "model": f"fallback ({type(exc).__name__})",
            "explanation": fallback,
        }
