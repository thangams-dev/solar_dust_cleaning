from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


STATUS_STYLES = {
    "disconnected": "critical",
    "unavailable": "critical",
    "failed": "critical",
    "crack": "critical",
    "critical": "critical",
    "connected": "good",
    "receiving data": "good",
    "port open": "good",
    "running": "good",
    "completed": "good",
    "healthy": "good",
    "normal": "good",
    "simulated": "info",
    "demo": "warning",
    "warning": "warning",
    "dust": "warning",
    "partial shading": "warning",
    "info": "info",
    "unknown": "neutral",
    "not configured": "neutral",
    "idle": "neutral",
    "awaiting": "neutral",
}


def status_badge(label: str) -> None:
    lowered = label.lower()
    style = next((value for key, value in STATUS_STYLES.items() if key in lowered), "neutral")
    st.markdown(
        f'<span class="status-badge status-{style}"><span class="status-dot"></span>'
        f"{label}</span>",
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(
        f'<div class="section-heading"><h3>{title}</h3>'
        f'{"<span>" + subtitle + "</span>" if subtitle else ""}</div>',
        unsafe_allow_html=True,
    )


def metric_cards(items: list[dict]) -> None:
    columns = st.columns(len(items), gap="medium")
    for column, item in zip(columns, items):
        with column:
            with st.container(border=True):
                icon, content = st.columns([0.22, 0.78], vertical_alignment="center")
                icon.markdown(
                    f'<div class="metric-icon">{item["icon"]}</div>',
                    unsafe_allow_html=True,
                )
                content.markdown(f'<div class="metric-label">{item["label"]}</div>',
                                 unsafe_allow_html=True)
                content.markdown(
                    f'<div class="metric-value">{item["value"]}'
                    f'<span class="metric-unit"> {item.get("unit", "")}</span></div>',
                    unsafe_allow_html=True,
                )
                if item.get("delta"):
                    content.caption(item["delta"])
                elif item.get("note"):
                    content.caption(item["note"])


def filter_readings(records: list[dict], period: str) -> list[dict]:
    windows = {
        "1 Hour": timedelta(hours=1),
        "6 Hours": timedelta(hours=6),
        "Today": timedelta(days=1),
        "7 Days": timedelta(days=7),
    }
    threshold = datetime.now(timezone.utc) - windows[period]
    filtered = []
    for record in records:
        try:
            stamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            if stamp >= threshold:
                filtered.append(record)
        except (ValueError, TypeError, KeyError):
            continue
    return filtered


def electrical_figure(records: list[dict]):
    frame = pd.DataFrame(records)
    if frame.empty:
        return None
    frame = frame.sort_values("timestamp")
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("Voltage (V)", "Current (mA)", "Power (W)"),
    )
    series = (
        ("voltage", "#2563EB", "V"),
        ("current", "#F59E0B", "mA"),
        ("power", "#16A34A", "W"),
    )
    for row, (column, color, unit) in enumerate(series, start=1):
        values = frame[column] * 1000 if column == "current" else frame[column]
        figure.add_trace(
            go.Scatter(x=frame["timestamp"], y=values, mode="lines",
                       name=unit, line=dict(color=color, width=2.4),
                       hovertemplate=f"%{{x}}<br>%{{y:.3f}} {unit}<extra></extra>"),
            row=row, col=1,
        )
        figure.update_yaxes(title_text=unit, row=row, col=1, fixedrange=False,
                            gridcolor="#E8EEF5", zeroline=False)
    figure.update_layout(
        height=430, margin=dict(l=12, r=18, t=48, b=14), showlegend=False,
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#475569", size=11),
    )
    figure.update_xaxes(showgrid=False, tickformat="%H:%M", row=3, col=1)
    for row in (1, 2):
        figure.update_xaxes(showticklabels=False, row=row, col=1)
    return figure


def chart_container(title: str, description: str | None = None):
    with st.container(border=True):
        section_header(title, description)
        return st.container()


def activity_timeline(records: list[dict], limit: int = 6) -> None:
    if not records:
        st.caption("System events will appear here as the dashboard runs.")
        return
    ordered = sorted(records, key=lambda item: item.get("timestamp", ""), reverse=True)[:limit]
    for event in ordered:
        label = event.get("alert_type") or event.get("condition") or event.get("status") or "Sensor update"
        severity = event.get("severity", "Info")
        source = event.get("source", event.get("mode", "System"))
        with st.container(border=True):
            left, right = st.columns([0.72, 0.28], vertical_alignment="center")
            left.markdown(f"**{label}**")
            left.caption(event.get("description", f"{source} · {severity}"))
            right.caption(event.get("timestamp", ""))
            right.caption(source)


def local_timestamp(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%b %d, %Y · %H:%M:%S")
    except (ValueError, TypeError):
        return value
