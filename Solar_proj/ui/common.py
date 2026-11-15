from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_alert(db, alert_type: str, severity: str, description: str, source: str) -> None:
    db.insert("alerts", {"timestamp": now_iso(), "alert_type": alert_type,
                         "severity": severity, "description": description,
                         "source": source})


def add_inspection(db, result: dict) -> int:
    inspection_id = db.insert("inspections", {
        "timestamp": result["timestamp"], "filename": result["filename"],
        "condition": result["condition"], "confidence": result["confidence"],
        "affected_area": result["affected_area"], "source": result["source"],
    })
    alert_map = {
        "Dust": ("Dust detected", "Warning", "Dust was detected; cleaning eligibility depends on the configured safety checks."),
        "Crack": ("Crack detected", "Critical", "Possible panel damage detected. Automatic cleaning is prohibited."),
        "Partial Shading": ("Partial shading detected", "Warning", "Partial shading detected. Automatic cleaning is prohibited."),
    }
    if result["condition"] in alert_map:
        alert_type, severity, description = alert_map[result["condition"]]
        add_alert(db, alert_type, severity, description, result["source"])
    return inspection_id


def dataframe(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def metric_row(items: list[tuple[str, str, str]]) -> None:
    from ui.components import metric_cards
    metric_cards([
        {"label": label, "value": value, "unit": "", "icon": "•", "note": help_text}
        for label, value, help_text in items
    ])


def plot_readings(records: list[dict], y: str, title: str, color: str):
    if not records:
        st.info("No readings stored yet. The dashboard will add readings while it is open.")
        return
    frame = dataframe(records).sort_values("timestamp")
    display_values = frame[y] * 1000 if y == "current" else frame[y]
    unit = {"voltage": "V", "current": "mA", "power": "W", "energy_wh": "Wh"}.get(y, "")
    figure = go.Figure(go.Scatter(
        x=frame["timestamp"], y=display_values, mode="lines",
        line=dict(color=color, width=2.2), name=title,
        hovertemplate=f"%{{x}}<br>%{{y:.3f}} {unit}<extra></extra>",
    ))
    figure.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        margin=dict(l=15, r=15, t=45, b=12), height=285,
        showlegend=False, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#64748B", size=11),
        xaxis=dict(showgrid=False, title=None, tickformat="%b %d · %H:%M"),
        yaxis=dict(title=unit, gridcolor="#E8EEF5", zeroline=False),
    )
    with st.container(border=True):
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def latest(db, table: str) -> dict | None:
    rows = db.records(table, limit=1)
    return rows[0] if rows else None


def simulated_cleaning_images(before: Image.Image | None) -> tuple[Image.Image, Image.Image]:
    if before is None:
        before = Image.new("RGB", (480, 300), "#17324d")
        draw_before = ImageDraw.Draw(before)
        for x in range(0, 480, 60):
            draw_before.line((x, 0, x, 300), fill="#69a3c4", width=2)
        for y in range(0, 300, 50):
            draw_before.line((0, y, 480, y), fill="#69a3c4", width=2)
        draw_before.text((14, 12), "DEMO PANEL IMAGE", fill="white")
    after = ImageEnhance.Brightness(before.convert("RGB")).enhance(1.08)
    ImageDraw.Draw(after).rectangle((0, 0, after.width, 30), fill="#14532d")
    ImageDraw.Draw(after).text((10, 8), "SIMULATED AFTER CLEANING", fill="white")
    return before.copy(), after
