"""
app/dashboard.py
================
Streamlit dashboard for Smart Grid DL.

Calls the FastAPI backend (default http://localhost:8000) for all inference.
Local CSV parsing is used only for the load heatmap.

Run:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import io
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Grid DL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global font: Josefin Sans via Google Fonts
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Josefin+Sans:ital,wght@0,100..700;1,100..700&display=swap');

    html, body, [class*="css"], .stApp,
    .stMarkdown, p, li, span, div,
    .stMetric label, .stMetric div,
    h1, h2, h3, h4, h5, h6,
    .stButton button, .stTextInput input,
    .stSelectbox select, .stFileUploader label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        font-family: 'Josefin Sans', sans-serif !important;
        letter-spacing: 0.01em;
    }

    h1 { font-weight: 700 !important; }
    h2, h3 { font-weight: 600 !important; }

    /* Metric card value — larger and bolder */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.75;
    }

    /* Section headers */
    .section-header {
        font-family: 'Josefin Sans', sans-serif !important;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.6;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LSTM_MAE_MW  = 72.9   # known test-set MAE — used for the forecast error band
_SMAPE_PCT    = 6.0    # known test-set sMAPE — static forecast-error penalty

_SEVERITY_COLOR: dict[str, str] = {
    "normal": "#2ecc71",
    "low":    "#f1c40f",
    "medium": "#e67e22",
    "high":   "#e74c3c",
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _parse_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes; set DatetimeIndex when a timestamp column exists."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp").sort_index()
    return df


def _call_full_analysis(api_url: str, file_bytes: bytes, filename: str) -> dict[str, Any] | None:
    """POST the CSV to /full-analysis; return JSON or display a friendly error."""
    url = f"{api_url.rstrip('/')}/full-analysis"
    try:
        resp = httpx.post(
            url,
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=180.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error(
            f"⚠️ Cannot reach the API at **{api_url}**.\n\n"
            "Start the backend with:\n"
            "```\nuvicorn api.main:app --reload --port 8000\n```"
        )
    except httpx.TimeoutException:
        st.error("⚠️ API request timed out (180 s). Try a smaller CSV.")
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"⚠️ API error {exc.response.status_code}: {detail}")
    return None


def _severity_label(error: float, threshold: float) -> str:
    """Classify a reconstruction error — mirrors predictor severity logic."""
    if error <= threshold:
        return "normal"
    excess = (error - threshold) / threshold
    if excess < 0.25:
        return "low"
    if excess < 0.75:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# KPI section
# ---------------------------------------------------------------------------


def _render_kpis(
    forecast: dict[str, Any],
    extended: dict[str, Any],
    anomalies: dict[str, Any],
) -> None:
    """Render six KPI metric cards in two rows of three."""
    preds_24  = forecast["predictions"]
    preds_168 = extended["predictions"]

    peak_load    = max(preds_24)
    avg_24       = round(sum(preds_24) / len(preds_24), 1)
    avg_168      = round(sum(preds_168) / len(preds_168), 1)
    n_anom       = anomalies["n_anomalies"]
    anomaly_rate = anomalies["anomaly_rate"]
    ghs          = round(max(0.0, 100.0 - anomaly_rate - _SMAPE_PCT), 1)

    st.markdown('<p class="section-header">Key Performance Indicators</p>', unsafe_allow_html=True)

    # Row 1
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak Load (MW)",        f"{peak_load:,.1f}")
    c2.metric("Anomalies Detected",    str(n_anom))
    c3.metric("sMAPE (%)",             f"{_SMAPE_PCT:.1f}")

    st.write("")   # breathing room between rows

    # Row 2
    c4, c5, c6 = st.columns(3)
    c4.metric(
        "Grid Health Score",
        f"{ghs:.1f} / 100",
        delta=f"{ghs - 100:.1f}" if ghs < 100 else "Optimal",
        delta_color="inverse",
    )
    c5.metric("Avg Forecast — Next 24 h (MW)",  f"{avg_24:,.1f}")
    c6.metric("Avg Forecast — Next 168 h (MW)", f"{avg_168:,.1f}")


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _chart_forecast_24(forecast: dict[str, Any]) -> go.Figure:
    """24-hour LSTM forecast with ±MAE shaded error band."""
    preds  = forecast["predictions"]
    labels = forecast["timestamps"]
    upper  = [p + _LSTM_MAE_MW for p in preds]
    lower  = [max(0.0, p - _LSTM_MAE_MW) for p in preds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=upper, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=lower, mode="lines",
        fill="tonexty", fillcolor="rgba(99,110,250,0.15)",
        line=dict(width=0), name=f"±{_LSTM_MAE_MW:.0f} MW band",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=preds,
        mode="lines+markers",
        line=dict(color="#636efa", width=2.5),
        marker=dict(size=5),
        name="Predicted load",
    ))
    fig.update_layout(
        title="24-Hour Load Forecast",
        xaxis_title="Time",
        yaxis_title="Load (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Josefin Sans, sans-serif"),
        height=400,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def _chart_forecast_168(extended: dict[str, Any]) -> go.Figure:
    """168-hour (7-day) iterative LSTM forecast."""
    preds  = extended["predictions"]
    labels = extended["timestamps"]

    # Smooth upper/lower band using a wider margin for longer horizons
    margin = _LSTM_MAE_MW * 1.5
    upper  = [p + margin for p in preds]
    lower  = [max(0.0, p - margin) for p in preds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=upper, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=lower, mode="lines",
        fill="tonexty", fillcolor="rgba(0,204,150,0.12)",
        line=dict(width=0), name=f"±{margin:.0f} MW band",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=preds,
        mode="lines",
        line=dict(color="#00cc96", width=2),
        name="168 h forecast",
    ))
    # Mark day boundaries
    for i in range(1, 7):
        fig.add_vline(
            x=labels[i * 24 - 1] if i * 24 - 1 < len(labels) else labels[-1],
            line_dash="dot", line_color="rgba(150,150,150,0.5)", line_width=1,
        )
    fig.update_layout(
        title="168-Hour (7-Day) Extended Load Forecast",
        xaxis_title="Time",
        yaxis_title="Load (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Josefin Sans, sans-serif"),
        height=400,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def _chart_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """Average load heatmap: hour-of-day (y) × day-of-week (x)."""
    if "load" not in df.columns:
        return None

    if "hour" in df.columns and "day_of_week" in df.columns:
        hm = df[["hour", "day_of_week", "load"]].copy()
    elif isinstance(df.index, pd.DatetimeIndex):
        hm = pd.DataFrame({
            "hour":        df.index.hour,
            "day_of_week": df.index.dayofweek,
            "load":        df["load"].values,
        })
    else:
        return None

    pivot = (
        hm.groupby(["hour", "day_of_week"])["load"]
        .mean()
        .unstack("day_of_week")
    )
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = [day_names[c] for c in pivot.columns if c < 7]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlBu_r",
        colorbar=dict(title="Avg MW"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Average Load — Hour of Day × Day of Week",
        xaxis_title="Day of week",
        yaxis_title="Hour of day",
        font=dict(family="Josefin Sans, sans-serif"),
        height=420,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def _chart_anomaly_timeline(anomalies: dict[str, Any]) -> go.Figure:
    """Scatter of all window reconstruction errors, coloured by severity."""
    errors    = anomalies["reconstruction_errors"]
    threshold = anomalies["threshold"]

    by_sev: dict[str, dict[str, list]] = {
        s: {"x": [], "y": []} for s in ["normal", "low", "medium", "high"]
    }
    for i, err in enumerate(errors):
        sev = _severity_label(err, threshold)
        by_sev[sev]["x"].append(i)
        by_sev[sev]["y"].append(err)

    fig = go.Figure()
    display = {"normal": "Normal", "low": "Low", "medium": "Medium", "high": "High"}
    sizes   = {"normal": 5, "low": 9, "medium": 9, "high": 10}
    opacs   = {"normal": 0.45, "low": 1.0, "medium": 1.0, "high": 1.0}

    for sev in ["normal", "low", "medium", "high"]:
        if by_sev[sev]["x"]:
            fig.add_trace(go.Scatter(
                x=by_sev[sev]["x"],
                y=by_sev[sev]["y"],
                mode="markers",
                name=display[sev],
                marker=dict(
                    color=_SEVERITY_COLOR[sev],
                    size=sizes[sev],
                    opacity=opacs[sev],
                ),
            ))

    fig.add_hline(
        y=threshold,
        line_dash="dash", line_color="#e74c3c",
        annotation_text=f"P99 threshold = {threshold:.4f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Anomaly Timeline — Reconstruction Error per 24-Hour Window",
        xaxis_title="Window index",
        yaxis_title="Reconstruction error (MAE)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Josefin Sans, sans-serif"),
        height=400,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> tuple[bytes | None, str, str, bool]:
    """Return (file_bytes, filename, api_url, run_clicked)."""
    with st.sidebar:
        st.title("⚡ Smart Grid DL")
        st.caption("Power Load Intelligence Platform")
        st.divider()

        uploaded = st.file_uploader(
            "Upload feature CSV",
            type=["csv"],
            help="Processed feature CSV — see column requirements below.",
        )

        api_url = st.text_input("API base URL", value="http://localhost:8000")

        run_clicked = st.button(
            "▶ Run Analysis",
            use_container_width=True,
            type="primary",
            disabled=(uploaded is None),
        )

        st.divider()
        st.markdown(
            "**Expected CSV columns**\n\n"
            "| Column | Notes |\n"
            "|--------|-------|\n"
            "| `timestamp` | optional — enables time axis |\n"
            "| `load` | MW, required for anomaly detection |\n"
            "| `temperature`, `humidity` | weather |\n"
            "| `is_weekend`, `is_holiday` | binary flags |\n"
            "| `hour_sin/cos` | cyclic encoding |\n"
            "| `dayofweek_sin/cos` | cyclic encoding |\n"
            "| `month_sin/cos` | cyclic encoding |\n"
            "| `lag_1`, `lag_24`, `lag_168` | load lags |\n"
            "| `rolling_mean_24`, `rolling_std_24` | 24 h stats |\n"
            "| `rolling_mean_168` | weekly mean |\n\n"
            "**Minimum 192 rows** required."
        )

    if uploaded is None:
        return None, "", api_url, run_clicked

    file_bytes = uploaded.read()
    return file_bytes, uploaded.name, api_url, run_clicked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point — called by Streamlit when the script runs."""
    file_bytes, filename, api_url, run_clicked = _render_sidebar()

    st.title("Smart Grid DL — Power Load Intelligence")
    st.caption("Upload a grid-feature CSV, click **▶ Run Analysis**, and explore the results.")

    if file_bytes is None:
        st.info("📂 Upload a feature CSV in the sidebar to get started.")
        return

    df = _parse_csv(file_bytes)
    st.success(f"CSV loaded — **{len(df):,}** rows × **{df.shape[1]}** columns")

    if not run_clicked:
        st.info("Click **▶ Run Analysis** in the sidebar to start inference.")
        return

    # ── API call ────────────────────────────────────────────────────────────
    with st.spinner("Running models via API …"):
        results = _call_full_analysis(api_url, file_bytes, filename)

    if results is None:
        return

    forecast  = results["forecast"]
    extended  = results["extended_forecast"]
    anomalies = results["anomalies"]

    # ── KPI cards ───────────────────────────────────────────────────────────
    st.divider()
    _render_kpis(forecast, extended, anomalies)

    # ── 24-hour forecast ────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">24-Hour Load Forecast</p>', unsafe_allow_html=True)
    st.plotly_chart(_chart_forecast_24(forecast), use_container_width=True)

    # ── 168-hour extended forecast ──────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">168-Hour Extended Forecast</p>', unsafe_allow_html=True)
    st.plotly_chart(_chart_forecast_168(extended), use_container_width=True)

    # ── Load heatmap ────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">Historical Load Pattern</p>', unsafe_allow_html=True)
    hm = _chart_heatmap(df)
    if hm is not None:
        st.plotly_chart(hm, use_container_width=True)
    else:
        st.info(
            "Heatmap requires a 'load' column plus 'hour' & 'day_of_week' columns "
            "or a datetime index."
        )

    # ── Anomaly timeline ────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">Anomaly Detection</p>', unsafe_allow_html=True)
    st.plotly_chart(_chart_anomaly_timeline(anomalies), use_container_width=True)

    # Flagged window detail
    n_anom = anomalies["n_anomalies"]
    if n_anom > 0:
        with st.expander(f"🚨 {n_anom} flagged window{'s' if n_anom != 1 else ''} — detail"):
            st.dataframe(
                pd.DataFrame({
                    "Timestamp": anomalies["flagged_timestamps"],
                    "Severity":  anomalies["severities"],
                }),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.success("✅ No anomalies detected in the uploaded data.")


if __name__ == "__main__":
    main()
