from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import streamlit as st

from ui.common import plot_readings
from ui.components import local_timestamp, metric_cards, section_header, status_badge


def render(db, config, serial_manager):
    source = "Demo (simulated)" if config.operating_mode == "Demo Mode" else "Hardware"
    st.caption(f"Data source · {source}. Other-mode history is not included in these live charts.")
    period = st.selectbox("Time range", ["Last 1 hour", "Last 6 hours", "Today", "Last 7 days",
                                        "Last 30 days", "Custom date range"])
    if period == "Custom date range":
        today = date.today()
        start = st.date_input("Start date", value=today - timedelta(days=6), key="live_start")
        end = st.date_input("End date", value=today, key="live_end")
        if start > end:
            st.error("Start date must be on or before end date.")
            return
        earliest = datetime.combine(start, time.min, tzinfo=timezone.utc).isoformat()
        ending = datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc).isoformat()
        records = db.records("readings", limit=10000, where="source = ? AND timestamp >= ? AND timestamp < ?",
                             params=(source, earliest, ending), descending=False)
    else:
        days = {"Last 1 hour": timedelta(hours=1), "Last 6 hours": timedelta(hours=6),
                "Today": timedelta(days=1), "Last 7 days": timedelta(days=7),
                "Last 30 days": timedelta(days=30)}[period]
        earliest = (datetime.now(timezone.utc) - days).isoformat()
        records = db.records("readings", limit=10000, where="source = ? AND timestamp >= ?",
                             params=(source, earliest), descending=False)
    if not records:
        if config.operating_mode == "Hardware Mode":
            status_badge("ESP32 DISCONNECTED" if not serial_manager.connected else "AWAITING SENSOR DATA")
            st.info("No hardware readings are available yet. Valid ESP32 telemetry will appear here when received.")
        else:
            st.info("No readings are available in this time range yet.")
        return
    latest = records[-1]
    metrics = []
    for label, key, unit in (("Voltage", "voltage", "V"), ("Current", "current", "A"),
                             ("Power", "power", "W"), ("Energy", "energy_wh", "Wh")):
        values = [float(item[key]) for item in records]
        display_latest = latest[key] * 1000 if key == "current" else latest[key]
        display_values = [value * 1000 for value in values] if key == "current" else values
        display_unit = "mA" if key == "current" else unit
        metrics.append({
            "label": label,
            "value": f"{display_latest:.2f}" if key != "energy_wh" else f"{display_latest:.4f}",
            "unit": display_unit,
            "icon": {"Voltage": "ϟ", "Current": "↗", "Power": "▰", "Energy": "☀"}[label],
            "note": (f"Min {min(display_values):.3f} · Max {max(display_values):.3f} · "
                     f"Avg {sum(display_values)/len(display_values):.3f} {display_unit}"),
        })
    metric_cards(metrics)
    sensor_status = ("Simulated" if config.operating_mode == "Demo Mode"
                     else st.session_state.get("hardware_status", {}).get("sensor", "Unavailable"))
    st.caption(f"Last updated · {local_timestamp(latest['timestamp'])} · Source · {latest['source']}")
    st.caption(f"INA219 status · {sensor_status}")
    section_header("Electrical trends", f"{period} · {len(records)} stored readings")
    for key, title, color in (
        ("voltage", "Voltage", "#2563EB"),
        ("current", "Current", "#F59E0B"),
        ("power", "Power", "#16A34A"),
        ("energy_wh", "Energy generated", "#7C3AED"),
    ):
        plot_readings(records, key, title, color)
    section_header("Recent readings", "Newest records first")
    frame = pd.DataFrame(records).sort_values("timestamp", ascending=False)
    st.dataframe(frame.head(100), width="stretch", hide_index=True)
