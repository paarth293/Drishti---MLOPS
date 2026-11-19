"""Unit tests for Drishti GenAI Explainer and router fallback mode."""

from app.copilot.explainer import explain_incident, _fallback_explanation


def test_copilot_fallback_explanation_empty():
    res = explain_incident(ranking=[])
    assert res["project"] == "Drishti"
    assert res["mode"] == "fallback"
    assert "No active root-cause candidates" in res["explanation"]


def test_copilot_fallback_explanation_with_ranking():
    ranking = [("auth-service", 0.85), ("payment-gateway", 0.62)]
    res = explain_incident(ranking=ranking, anomalies=[{"ts": "2026-01-01T00:00:00", "y": 99.0}])
    assert res["project"] == "Drishti"
    assert "auth-service" in res["explanation"]
    assert "payment-gateway" in res["explanation"]
