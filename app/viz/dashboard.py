"""Streamlit dashboard for AIOps Copilot."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pyvis.network import Network

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _fetch_json(path: str, params: dict[str, Any | None] = None) -> Any:
    """Fetch JSON from the API with basic error handling."""

    try:
        response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        st.error(f"API request failed: {error}")
        return {}


def _parse_metrics_text(metrics_text: str) -> dict[str, float]:
    """Parse Prometheus-style metrics into a dictionary."""

    metrics: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            name, value = line.split()
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def _fetch_measurements_df(service_id: str, limit: int, since: datetime | None = None) -> pd.DataFrame:
    """Return measurements for a service as a dataframe."""

    params: dict[str, Any] = {"service_id": service_id, "limit": limit}
    if since is not None:
        params["since"] = since.isoformat()
    payload = _fetch_json("/measurements", params=params)
    if not payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def _fetch_anomalies_df(service_id: str, limit: int, since: datetime | None = None) -> pd.DataFrame:
    """Return anomalies for a service as a dataframe."""

    params: dict[str, Any] = {"service_id": service_id, "limit": limit}
    if since is not None:
        params["since"] = since.isoformat()
    payload = _fetch_json("/anomalies", params=params)
    if not payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def overview_page(services: list[str]) -> None:
    """Render the overview page with metrics and recent anomalies."""

    st.header("Drishti Overview")

    try:
        metrics_response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        metrics_response.raise_for_status()
        metrics_text = metrics_response.text
    except requests.RequestException as error:
        st.error(f"Unable to load metrics: {error}")
        metrics_text = ""
    metrics_values = _parse_metrics_text(metrics_text)

    ingested = int(metrics_values.get("aiops_copilot_ingested_points", 0))
    anomalies = int(metrics_values.get("aiops_copilot_anomaly_count", 0))
    latency = metrics_values.get("aiops_copilot_last_detection_latency_seconds", 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Ingested Points", f"{ingested:,}")
    c2.metric("Detected Anomalies", f"{anomalies:,}")
    c3.metric("Last Detection Latency (s)", f"{latency:.3f}")

    with st.expander("Raw metrics payload", expanded=False):
        st.code(metrics_text or "No metrics available.", language="text")

    graph_payload = _fetch_json("/graph")
    st.subheader("Active Services")
    if graph_payload:
        st.write(", ".join(graph_payload.get("nodes", [])))

    st.subheader("Top Suspects (RCA)")
    if services:
        topk = _fetch_json("/rca/topk", params={"k": min(len(services), 5)})
        if topk and topk.get("ranking"):
            topk_df = pd.DataFrame(topk["ranking"], columns=["Service", "Score"])
            st.table(topk_df)
        else:
            st.info("No RCA ranking available yet. Run detections to populate it.")

    st.subheader("Latest Anomalies")
    if services:
        recent_rows = []
        for svc in services[:5]:
            df = _fetch_anomalies_df(svc, limit=5)
            if df.empty:
                continue
            latest = df.sort_values("ts", ascending=False).head(5).copy()
            latest.insert(0, "service_id", svc)
            recent_rows.append(latest)
        if recent_rows:
            recent_df = pd.concat(recent_rows, ignore_index=True)
            recent_df = recent_df[["service_id", "ts", "y", "y_hat", "residual", "q_alpha", "is_anomaly"]]
            st.dataframe(
                recent_df.sort_values("ts", ascending=False),
                width="stretch",
                height=300,
            )
        else:
            st.info("No anomalies recorded yet. Trigger detection to populate this table.")


def service_page(service_id: str, limit: int, window_hours: int, show_residuals: bool) -> None:
    """Render the service page with anomaly timeline."""

    st.header(f"Service Details - {service_id}")
    since = datetime.utcnow() - timedelta(hours=window_hours) if window_hours > 0 else None
    measurements = _fetch_measurements_df(service_id, limit=limit, since=since)
    if measurements.empty:
        st.info("No measurements available for this service yet.")
        return

    anomalies = _fetch_anomalies_df(service_id, limit=limit, since=since)
    measurement_frame = measurements.set_index("ts")
    if anomalies.empty:
        anomaly_frame = pd.DataFrame(columns=["y_hat", "residual", "q_alpha", "is_anomaly"]).set_index(
            pd.Index([], name="ts")
        )
    else:
        cleaned = anomalies.drop_duplicates(subset="ts", keep="last").set_index("ts")
        anomaly_frame = cleaned[["y_hat", "residual", "q_alpha", "is_anomaly"]]
    combined = measurement_frame.join(anomaly_frame, how="left")
    combined["is_anomaly"] = combined["is_anomaly"].astype("boolean").fillna(False).astype(bool)
    combined = combined.reset_index()

    total_points = len(combined)
    anomalies_flagged = int(combined["is_anomaly"].sum())
    last_anomaly_ts = combined.loc[combined["is_anomaly"], "ts"].max() if anomalies_flagged else None
    ratio = anomalies_flagged / total_points if total_points else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations Shown", total_points)
    c2.metric("Anomaly Records", anomalies_flagged)
    c3.metric("Anomaly Share", f"{ratio:.1%}")
    c4.metric(
        "Last Anomaly",
        last_anomaly_ts.strftime("%Y-%m-%d %H:%M:%S") if last_anomaly_ts is not None else "N/A",
    )

    observed_fig = go.Figure()
    observed_fig.add_trace(
        go.Scatter(
            x=combined["ts"],
            y=combined["y"],
            mode="lines",
            name="Observed",
        )
    )

    if combined["y_hat"].notna().any():
        y_hat_series = pd.to_numeric(combined["y_hat"], errors="coerce").ffill().bfill()
        observed_fig.add_trace(go.Scatter(x=combined["ts"], y=y_hat_series, mode="lines", name="Predicted"))
        if combined["q_alpha"].notna().any():
            band = pd.to_numeric(combined["q_alpha"], errors="coerce").ffill().bfill()
            upper = y_hat_series + band
            lower = y_hat_series - band
            observed_fig.add_trace(
                go.Scatter(
                    x=list(combined["ts"]) + list(combined["ts"][::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.12)",
                    line=dict(color="rgba(255,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Conformal Band",
                )
            )

    anomalies_only = combined[combined["is_anomaly"]]
    if not anomalies_only.empty:
        observed_fig.add_trace(
            go.Scatter(
                x=anomalies_only["ts"],
                y=anomalies_only["y"],
                mode="markers",
                marker=dict(color="red", size=9),
                name="Anomalies",
            )
        )

    observed_fig.update_layout(xaxis_title="Timestamp", yaxis_title="Observed value", height=450)

    if show_residuals and anomalies_flagged:
        residual_fig = go.Figure()
        residual_fig.add_trace(
            go.Scatter(
                x=anomalies_only["ts"],
                y=anomalies_only["residual"],
                mode="lines+markers",
                name="Residual",
                line=dict(color="orange"),
            )
        )
        if anomalies_only["q_alpha"].notna().any():
            residual_fig.add_trace(
                go.Scatter(
                    x=anomalies_only["ts"],
                    y=anomalies_only["q_alpha"],
                    mode="lines",
                    name="+qα",
                    line=dict(color="rgba(255,0,0,0.5)", dash="dash"),
                )
            )
            residual_fig.add_trace(
                go.Scatter(
                    x=anomalies_only["ts"],
                    y=-anomalies_only["q_alpha"],
                    mode="lines",
                    name="-qα",
                    line=dict(color="rgba(255,0,0,0.5)", dash="dash"),
                )
            )
        residual_fig.update_layout(xaxis_title="Timestamp", yaxis_title="Residual", height=350)
        st.plotly_chart(residual_fig, config={"responsive": True})

    st.plotly_chart(observed_fig, config={"responsive": True})
    st.subheader("Detailed Records")
    st.dataframe(
        combined.sort_values("ts", ascending=False),
        width="stretch",
        height=400,
    )


def compare_page(services: list[str]) -> None:
    """Render a comparison view across multiple services."""

    st.header("Service Comparison")
    if not services:
        st.info("No services available. Run detection to populate data.")
        return

    default_selection = services[: min(len(services), 3)]
    selected = st.multiselect("Services", options=services, default=default_selection)
    if not selected:
        st.warning("Select at least one service to compare.")
        return

    measurement_frames: dict[str, pd.DataFrame] = {}
    start_bounds: list[pd.Timestamp] = []
    end_bounds: list[pd.Timestamp] = []
    for service in selected:
        df = _fetch_measurements_df(service, limit=5000)
        if df.empty:
            st.info(f"No measurements available for {service}.")
            continue
        df = df.sort_values("ts")
        measurement_frames[service] = df
        start_bounds.append(df["ts"].min())
        end_bounds.append(df["ts"].max())

    if not measurement_frames:
        st.info("No data returned for the selected services.")
        return

    overlapping_start = max(start_bounds)
    overlapping_end = min(end_bounds)
    if overlapping_start >= overlapping_end:
        st.warning("Selected services do not share an overlapping time window.")
        return

    min_bound = min(start_bounds)
    max_bound = max(end_bounds)
    start_default = overlapping_start.to_pydatetime()
    end_default = overlapping_end.to_pydatetime()

    start_ts, end_ts = st.slider(
        "Common period",
        min_value=min_bound.to_pydatetime(),
        max_value=max_bound.to_pydatetime(),
        value=(start_default, end_default),
    )
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)

    compare_fig = go.Figure()
    summary_rows = []
    for service, df in measurement_frames.items():
        measurements = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]
        if measurements.empty:
            continue
        compare_fig.add_trace(
            go.Scatter(
                x=measurements["ts"],
                y=measurements["y"],
                mode="lines",
                name=service,
            )
        )

        anomalies = _fetch_anomalies_df(service, limit=1000, since=start_ts.to_pydatetime())
        if not anomalies.empty:
            anomalies = anomalies[(anomalies["ts"] >= start_ts) & (anomalies["ts"] <= end_ts)]
        anomalies_flagged = int(anomalies["is_anomaly"].sum()) if not anomalies.empty else 0
        severity = None
        if not anomalies.empty and {"residual", "q_alpha"}.issubset(anomalies.columns):
            mask = anomalies["q_alpha"].abs() > 0
            normalised = (anomalies["residual"].abs() / anomalies["q_alpha"].abs()).where(mask)
            severity = float(normalised.mean()) if anomalies_flagged else 0.0

        summary_rows.append(
            {
                "service_id": service,
                "points": len(measurements),
                "anomalies": anomalies_flagged,
                "avg_severity": severity if severity is not None else "n/a",
                "window_start": measurements["ts"].min(),
                "window_end": measurements["ts"].max(),
            }
        )

    if not compare_fig.data:
        st.info("No data returned for the selected services.")
        return

    compare_fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Observed value",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(compare_fig, config={"responsive": True})

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values("anomalies", ascending=False)
        st.subheader("Comparison Summary")
        st.dataframe(summary_df, width="stretch")


def graph_page() -> None:
    """Render the dependency graph and RCA ranking."""

    st.header("Service Graph & RCA")
    graph_payload = _fetch_json("/graph")
    if not graph_payload:
        st.info("Graph is unavailable.")
        return
    network = Network(height="600px", width="100%", directed=True)
    for node in graph_payload.get("nodes", []):
        network.add_node(node, label=node)
    for edge in graph_payload.get("edges", []):
        network.add_edge(edge["source"], edge["target"], value=edge.get("weight", 1.0))
    st.subheader("Dependency Graph")
    st.components.v1.html(network.generate_html(), height=600, scrolling=True)

    st.subheader("Top-K RCA")
    topk = _fetch_json("/rca/topk", params={"k": 10})
    if topk and topk.get("ranking"):
        st.table(pd.DataFrame(topk["ranking"], columns=["Service", "Score"]))
    else:
        st.info("RCA ranking is empty. Run detection to populate scores.")



    st.subheader("Drishti AI Incident Copilot (LangChain + Gemini)")
    if st.button("Explain Incident with Drishti AI"):
        with st.spinner("Generating root-cause explanation..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/rca/explain",
                    json={"k": 5},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                st.caption(f"Mode: `{data.get('mode')}` | Model: `{data.get('model')}`")
                st.markdown(data.get("explanation", "No explanation returned."))
            except Exception as exc:
                st.error(f"Could not reach /rca/explain: {exc}")


def main() -> None:
    """Entry point for the dashboard."""

    st.set_page_config(page_title="Drishti — AIOps & MLOps Copilot", layout="wide")
    st.sidebar.title("Drishti")
    graph_payload = _fetch_json("/graph")
    services = graph_payload.get("nodes", []) if graph_payload else []
    page = st.sidebar.radio("Page", options=["Overview", "Service", "Compare", "Graph"])
    if page == "Overview":
        overview_page(services)
    elif page == "Service":
        if not services:
            st.info("No services available. Run detection to populate data.")
            return
        service_id = st.sidebar.selectbox("Service", options=services)
        limit = st.sidebar.slider("Samples per view", min_value=50, max_value=1000, value=300, step=50)
        window_hours = st.sidebar.slider("Lookback window (hours)", min_value=0, max_value=168, value=0, step=6)
        show_residuals = st.sidebar.checkbox("Show residuals view", value=False)
        service_page(service_id, limit=limit, window_hours=window_hours, show_residuals=show_residuals)
    elif page == "Compare":
        compare_page(services)
    else:
        graph_page()


if __name__ == "__main__":
    main()
