"""Launch the full NAB showcase: API, ingest/detect pipeline, optional Streamlit."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests  # noqa: E402

from scripts.apply_graph import apply_graph as send_graph_payload  # noqa: E402
from scripts.apply_graph import load_graph as load_graph_config  # noqa: E402
from scripts.run_nab_showcase import SUMMARY_PATH, run_showcase  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full NAB showcase end-to-end.")
    parser.add_argument("--uvicorn-host", default="127.0.0.1")
    parser.add_argument("--uvicorn-port", type=int, default=8000)
    parser.add_argument("--uvicorn-workers", type=int, default=0, help="0 = auto (CPU count).")
    parser.add_argument(
        "--no-start-api",
        action="store_true",
        help="Skip starting uvicorn (expects API already running at --api-url).",
    )
    parser.add_argument("--api-url", help="Override API base URL (default http://host:port).")
    parser.add_argument(
        "--launch-streamlit",
        action="store_true",
        help="Launch the Streamlit dashboard after the showcase run.",
    )
    parser.add_argument("--streamlit-app", default="app/viz/dashboard.py")
    parser.add_argument("--streamlit-port", type=int, default=8501)
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="Do not keep processes alive after the showcase is complete.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=90,
        help="Seconds to wait for the API health endpoint before aborting.",
    )

    # NAB showcase tuning flags (mirror run_nab_showcase)
    parser.add_argument("--data-dir", default=str(SUMMARY_PATH.parent.parent / "data" / "nab"))
    parser.add_argument("--detector", default="baseline", choices=["forecast", "baseline"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--services", nargs="+")
    parser.add_argument("--max-workers", type=int, default=0, help="0 = auto (CPU count).")
    parser.add_argument("--anomaly-limit", type=int, default=50)
    parser.add_argument("--rca-topk", type=int, default=5)
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH))
    parser.add_argument("--report-service")
    parser.add_argument("--report-window-ref", type=int, default=500)
    parser.add_argument("--report-window-cur", type=int, default=200)
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument(
        "--graph-file",
        default=str(Path("configs/graphs/nab.yaml")),
        help="Graph YAML file to apply once the API is up.",
    )
    parser.add_argument(
        "--skip-apply-graph",
        action="store_true",
        help="Disable automatic graph application.",
    )
    return parser.parse_args()


def _auto_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _wait_for_api(api_url: str, timeout: int) -> None:
    health_url = f"{api_url}/health"
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    time.sleep(1.0)
    while time.monotonic() < deadline:
        try:
            response = requests.get(health_url, timeout=5)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.5)
    reason = f" (last error: {last_error})" if last_error else ""
    raise RuntimeError(f"API not reachable at {health_url} after {timeout} seconds{reason}.")


def _start_subprocess(cmd: list[str], env: dict[str, str | None] = None) -> subprocess.Popen[Any]:
    return subprocess.Popen(cmd, env=env or os.environ.copy())  # noqa: S603


def _terminate_process(proc: subprocess.Popen[Any | None], timeout: int = 10) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> None:
    args = parse_args()
    api_url = args.api_url.rstrip("/") if args.api_url else f"http://{args.uvicorn_host}:{args.uvicorn_port}"
    uvicorn_process: subprocess.Popen[Any | None] = None
    streamlit_process: subprocess.Popen[Any | None] = None
    python_executable = sys.executable

    try:
        if not args.no_start_api:
            uvicorn_workers = args.uvicorn_workers or _auto_cpu_count()
            uvicorn_cmd = [
                python_executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                args.uvicorn_host,
                "--port",
                str(args.uvicorn_port),
                "--workers",
                str(uvicorn_workers),
            ]
            uvicorn_process = _start_subprocess(uvicorn_cmd)
            print(f"[info] Started uvicorn with PID {uvicorn_process.pid} using {uvicorn_workers} workers.")

        _wait_for_api(api_url, timeout=args.startup_timeout)
        print(f"[info] API reachable at {api_url}")

        if not args.skip_apply_graph and args.graph_file:
            try:
                graph_payload = load_graph_config(Path(args.graph_file))
                send_graph_payload(api_url, graph_payload)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Failed to apply graph {args.graph_file}: {exc}")
            else:
                print(f"[info] Applied graph from {args.graph_file}")

        run_showcase(
            api_url=api_url,
            data_dir=args.data_dir,
            services=args.services,
            detector=args.detector,
            alpha=args.alpha,
            timeout=args.timeout,
            chunk_size=args.chunk_size,
            max_workers=args.max_workers,
            anomaly_limit=args.anomaly_limit,
            rca_topk=args.rca_topk,
            summary_path=args.summary_path,
            report_service=args.report_service,
            report_window_ref=args.report_window_ref,
            report_window_cur=args.report_window_cur,
            skip_report=args.skip_report,
            verbose=True,
        )

        if args.launch_streamlit:
            streamlit_cmd = [
                python_executable,
                "-m",
                "streamlit",
                "run",
                args.streamlit_app,
                "--server.port",
                str(args.streamlit_port),
                "--server.headless",
                "true",
            ]
            streamlit_env = os.environ.copy()
            streamlit_env["API_BASE_URL"] = api_url
            streamlit_process = _start_subprocess(streamlit_cmd, env=streamlit_env)
            print(
                f"[info] Started Streamlit dashboard (PID {streamlit_process.pid}) "
                f"-> http://localhost:{args.streamlit_port}"
            )

        if args.no_hold:
            return

        print("\n[info] Showcase complete. Press Ctrl+C to stop the services.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[info] Shutting down processes...")
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        _terminate_process(streamlit_process)
        _terminate_process(uvicorn_process)


if __name__ == "__main__":
    main()
