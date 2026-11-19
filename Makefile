PYTHON ?= python

.PHONY: setup lint fmt test run-api streamlit seed-s5 simulate seed-nab nab-detect nab-showcase full-showcase graph-apply graph-nab graph-default report docker-up docker-down

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check .
	black --check .
	isort --check-only .

fmt:
	black .
	isort .

test:
	pytest -q --cov=app --cov-report=term-missing

UVICORN_HOST ?= 127.0.0.1
UVICORN_PORT ?= 8000
UVICORN_WORKERS ?= 1
FULL_UVICORN_WORKERS ?= 0
FULL_STREAMLIT_PORT ?= 8501
FULL_LAUNCH_STREAMLIT ?= 1
FULL_NO_HOLD ?=
FULL_STARTUP_TIMEOUT ?= 90

run-api:
	uvicorn app.main:app --host $(UVICORN_HOST) --port $(UVICORN_PORT) $(if $(filter 1,$(UVICORN_WORKERS)),--reload,--workers $(UVICORN_WORKERS))

streamlit:
	streamlit run app/viz/dashboard.py

seed-s5:
	$(PYTHON) scripts/seed_yahoo_s5.py

simulate:
	$(PYTHON) scripts/simulate_services.py

seed-nab:
	$(PYTHON) scripts/seed_nab.py

DETECTOR ?= baseline
TIMEOUT ?= 300
CHUNK_SIZE ?= 500
SERIES ?=
NAB_MAX_WORKERS ?=
NAB_SUMMARY_PATH ?= reports/nab_summary.json
NAB_REPORT_SERVICE ?=
NAB_SKIP_REPORT ?=
NAB_ANOMALY_LIMIT ?= 50
NAB_RCA_TOPK ?= 5
NAB_REPORT_REF ?= 500
NAB_REPORT_CUR ?= 200

nab-detect:
	$(PYTHON) scripts/run_nab_ingest.py --detector $(DETECTOR) --timeout $(TIMEOUT) --chunk-size $(CHUNK_SIZE) $(if $(SERIES),--services $(SERIES),) $(if $(NAB_MAX_WORKERS),--max-workers $(NAB_MAX_WORKERS),)

nab-showcase:
	$(PYTHON) scripts/run_nab_showcase.py --detector $(DETECTOR) --timeout $(TIMEOUT) --chunk-size $(CHUNK_SIZE) $(if $(SERIES),--services $(SERIES),) $(if $(NAB_MAX_WORKERS),--max-workers $(NAB_MAX_WORKERS),) $(if $(NAB_SUMMARY_PATH),--summary-path $(NAB_SUMMARY_PATH),) $(if $(NAB_REPORT_SERVICE),--report-service $(NAB_REPORT_SERVICE),) $(if $(NAB_SKIP_REPORT),--skip-report,)

full-showcase:
	$(PYTHON) scripts/run_full_showcase.py --uvicorn-host $(UVICORN_HOST) --uvicorn-port $(UVICORN_PORT) --uvicorn-workers $(FULL_UVICORN_WORKERS) --startup-timeout $(FULL_STARTUP_TIMEOUT) $(if $(FULL_LAUNCH_STREAMLIT),--launch-streamlit,) --streamlit-port $(FULL_STREAMLIT_PORT) $(if $(FULL_NO_HOLD),--no-hold,) --detector $(DETECTOR) --timeout $(TIMEOUT) --chunk-size $(CHUNK_SIZE) $(if $(SERIES),--services $(SERIES),) $(if $(NAB_MAX_WORKERS),--max-workers $(NAB_MAX_WORKERS),) --anomaly-limit $(NAB_ANOMALY_LIMIT) --rca-topk $(NAB_RCA_TOPK) --summary-path $(NAB_SUMMARY_PATH) $(if $(NAB_REPORT_SERVICE),--report-service $(NAB_REPORT_SERVICE),) --report-window-ref $(NAB_REPORT_REF) --report-window-cur $(NAB_REPORT_CUR) $(if $(NAB_SKIP_REPORT),--skip-report,)

GRAPH_FILE ?= configs/graphs/nab.yaml

graph-apply:
	$(PYTHON) scripts/apply_graph.py --graph $(GRAPH_FILE)

graph-nab:
	$(MAKE) graph-apply GRAPH_FILE=configs/graphs/nab.yaml

graph-default:
	$(MAKE) graph-apply GRAPH_FILE=configs/graphs/microservices.yaml

report:
	$(PYTHON) scripts/run_evidently_report.py --service-id checkout

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
