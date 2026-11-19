# Drishti — 5-Minute Setup & Viva Cheat Sheet

## Step 1: Clone and Set Up Drishti

```bash
git clone https://github.com/paarth293/Drishti---MLOPS.git Drishti
cd Drishti
```

This repository is fully configured with all end-to-end components out of the box:
1. `app/copilot/explainer.py` & `app/copilot/router.py`: LangChain LCEL + Gemini (`/rca/explain`) with deterministic fallback.
2. `scripts/mlflow_tracking.py`: MLflow experiment tracking + `IsolationForest` `@champion` model registry.
3. `scripts/run_evidently_report.py`: Evidently AI drift-triggered auto-retraining feedback loop.
4. Full observability stack with Prometheus `/metrics`, Grafana provisioning, and Streamlit visualization dashboard.

## Step 2: Run Locally Before Your Exam

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -e .
python scripts/mlflow_tracking.py
```

Open 3 terminals:
- **Terminal 1 (FastAPI):** `uvicorn app.main:app --reload --port 8000`
- **Terminal 2 (Streamlit):** `streamlit run app/viz/dashboard.py`
- **Terminal 3 (MLflow UI):** `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`

## Step 3: 5-Minute Live Demo Flow for the Examiner

1. **Unit 1 (AIOps Anomaly Detection & RCA):** Open `http://localhost:8501` -> **Service** tab (show anomalies on time series) -> **Graph** tab (show NetworkX/PyVis dependency graph + Top-K RCA table).
2. **Unit 3 (FastAPI + GenAI Copilot):** Click **"Explain Incident with Drishti AI"** in the Graph tab (or call `POST /rca/explain` in `http://localhost:8000/docs`). Explain that it uses LangChain LCEL (`ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`) with a safe offline fallback.
3. **Unit 2 & 3 (MLflow Tracking & Model Registry):** Open `http://localhost:5000` -> show the `drishti-mlops-monitoring` experiment runs, logged metrics/artifacts, and the registered `drishti-anomaly-detector` model with the `@champion` alias.
4. **Unit 4 (Observability, Drift & Auto-Retraining):** Show `http://localhost:8000/metrics` (Prometheus exposition), `.github/workflows/ci.yml` (GitHub Actions CI), and run:
   ```bash
   python scripts/run_evidently_report.py --service-id ambient_temperature_system_failure --force-retrain
   ```
   Open `reports/latest.html` in a browser and refresh MLflow UI to show the new auto-retrained run!
