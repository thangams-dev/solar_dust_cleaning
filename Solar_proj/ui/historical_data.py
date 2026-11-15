from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import streamlit as st

from ui.common import plot_readings
from ui.components import local_timestamp, section_header
from utils.exporters import records_to_csv


def _range_filter(label: str) -> tuple[str, str]:
    today = date.today()
    if label == "Today":
        start, end = today, today
    elif label == "Yesterday":
        start = end = today - timedelta(days=1)
    elif label == "Last 7 days":
        start, end = today - timedelta(days=6), today
    elif label == "Last 30 days":
        start, end = today - timedelta(days=29), today
    else:
        start = st.date_input("Start date", value=today - timedelta(days=6), key="history_start")
        end = st.date_input("End date", value=today, key="history_end")
        if start > end:
            st.error("Start date must be on or before end date.")
            return "", ""
    beginning = datetime.combine(start, time.min, tzinfo=timezone.utc).isoformat()
    ending = datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc).isoformat()
    return beginning, ending


def render(db, config):
    period = st.selectbox("Filter records", ["Today", "Yesterday", "Last 7 days",
                                              "Last 30 days", "Custom date range"])
    source_choice = st.selectbox(
        "Data source", ["All sources", "Demo / simulated", "Hardware / real"]
    )
    start, end = _range_filter(period)
    if not start or not end:
        return
    tables = [("Electrical readings", "readings"), ("AI inspection history", "inspections"),
              ("Cleaning history", "cleanings")]
    for label, table in tables:
        where = "timestamp >= ? AND timestamp < ?"
        params = (start, end)
        if source_choice == "Demo / simulated":
            where += " AND source LIKE 'Demo%'"
        elif source_choice == "Hardware / real":
            where += " AND source IN ('Hardware', 'Real AI')"
        records = db.records(table, limit=10000, where=where,
                             params=params, descending=False)
        section_header(label)
        if table == "readings":
            for key, title, color in (
                ("voltage", "Voltage", "#2563EB"),
                ("current", "Current", "#F59E0B"),
                ("power", "Power", "#16A34A"),
                ("energy_wh", "Energy", "#7C3AED"),
            ):
                plot_readings(records, key, title, color)
        if records:
            frame = pd.DataFrame(records)
            if "timestamp" in frame:
                frame["timestamp"] = frame["timestamp"].map(local_timestamp)
            st.dataframe(frame, width="stretch", hide_index=True)
            st.download_button(f"Export {label} CSV", records_to_csv(records),
                               file_name=f"{table}.csv", mime="text/csv", key=f"export-{table}")
        else:
            st.info(f"No {label.lower()} in this date range.")
