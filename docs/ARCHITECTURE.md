# Drishti System Architecture & MLOps Pipeline Specification

## 1. System Overview
Drishti integrates four core pillars:
1. **AIOps Real-time Telemetry & Ingestion:** Ingests metric streams, microservice logs, and Numenta Anomaly Benchmark (NAB) time-series data.
2. **Anomaly Detection & Graph-based RCA:** Evaluates streaming metrics against statistical baselines, conformal prediction intervals, and online ARIMA models. When anomalies are detected, a directed NetworkX service dependency graph applies Personalized PageRank to rank suspect root causes.
3. **MLOps Lifecycle & Champion Model Registry:** Uses MLflow for experiment tracking, parameter logging, metric charting, artifact persistence, and manages an IsolationForest anomaly detector under the `@champion` model registry alias.
4. **Data Drift Detection & Closed-Loop Auto-Retraining:** Computes continuous statistical drift metrics using Evidently AI presets. When drift exceeds tolerance, automated retraining is triggered to promote newly calibrated models into MLflow.
5. **GenAI Copilot Incident Explainer:** Leverages LangChain Expression Language (LCEL) with Google Gemini (`prompt | model | parser`) to synthesize plain-English root-cause explanations and actionable remediation steps, complete with a deterministic offline fallback.

## 2. API & Service Topology
- **FastAPI Core:** `http://localhost:8000` (`/docs` for Swagger UI, `/metrics` for Prometheus exposition).
- **Streamlit Control Center:** `http://localhost:8501` (Service visualizer, PyVis interactive dependency graph, incident explainer).
- **MLflow Tracking Server:** `http://localhost:5000` (Runs, metric charts, model registry).
