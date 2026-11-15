import sqlite3
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from ui.components import local_timestamp, section_header, status_badge


def _database_status(db) -> tuple[str, str]:
    try:
        with db.connection() as connection:
            connection.execute("SELECT 1")
        return "Connected", "SQLite query succeeded"
    except sqlite3.Error as error:
        return "Unavailable", str(error)


def _device_card(name: str, status: str, detail: str, operating_mode: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{name}**")
        status_badge(status)
        st.caption(detail)
        st.caption(f"Mode · {operating_mode}")


def render(db, config, serial_manager):
    st.session_state.last_status_refresh = datetime.now().astimezone().isoformat()
    hardware = config.operating_mode == "Hardware Mode"
    statuses = st.session_state.get("hardware_status", {})
    esp = "Serial port open · device unverified" if hardware and serial_manager.connected else (
        "Disconnected" if hardware else "Simulated"
    )
    sensor = statuses.get("sensor", "Unavailable") if hardware else statuses.get("sensor", "Simulated")
    servo = statuses.get("servo", "Unavailable") if hardware else "Simulated"
    camera = "Active" if st.session_state.get("camera_active") else "Not checked"
    database_status, database_detail = _database_status(db)
    model_path = st.session_state.get("model_path", "")
    ai_status = "Not Configured"
    ai_detail = "No trained model loaded"
    if model_path and Path(model_path).is_file():
        ai_detail = "Model file is configured; inference loader is not connected"
    elif not hardware:
        ai_status = "Simulated"
        ai_detail = "Demo AI simulator · not a trained model"

    section_header("Device health", "Status is sourced from the current operating mode and reported device state.")
    components = [
        ("ESP32", esp, f"Serial port · {config.serial_port or 'Not configured'}"),
        ("INA219", sensor, "Voltage/current telemetry over ESP32 serial"),
        ("USB webcam", camera, f"Device index · {config.camera_index}"),
        ("AI model", ai_status, ai_detail),
        ("Servo motor", servo, "Cleaning status reported by ESP32" if hardware else "Demo actuator · no physical movement"),
        ("Database", database_status, database_detail),
        ("Python application", "Running", "Streamlit dashboard process"),
    ]
    columns = st.columns(4, gap="medium")
    for column, component in zip(columns * 2, components):
        with column:
            _device_card(component[0], component[1], component[2], config.operating_mode)

    section_header("Connection details")
    info, updates = st.columns(2, gap="medium")
    with info:
        with st.container(border=True):
            st.markdown("**Operating configuration**")
            status_badge("HARDWARE MODE" if hardware else "DEMO MODE")
            st.write(f"Serial port · `{config.serial_port or 'Not configured'}`")
            st.write(f"Baud rate · `{config.baud_rate}`")
            st.write(f"Last received message · `{serial_manager.last_message}`")
            if serial_manager.last_error:
                st.warning(f"Serial status · {serial_manager.last_error}")
    with updates:
        with st.container(border=True):
            st.markdown("**Recent updates**")
            source = "Demo (simulated)" if not hardware else "Hardware"
            readings = db.records("readings", limit=1, where="source = ?", params=(source,))
            st.write(f"Last sensor update · {local_timestamp(readings[0]['timestamp']) if readings else '—'}")
            st.write(f"Last webcam frame · {local_timestamp(st.session_state.get('camera_last_capture'))}")
            uptime_seconds = max(0, int(time.monotonic() - st.session_state.started_at))
            st.write(
                f"Application uptime · {uptime_seconds // 3600}h "
                f"{(uptime_seconds % 3600) // 60}m {uptime_seconds % 60}s"
            )
            st.write(f"Status refreshed · {local_timestamp(st.session_state.get('last_status_refresh'))}")

    if st.button("Refresh Status", type="secondary"):
        st.session_state.last_status_refresh = datetime.now().astimezone().isoformat()
        st.rerun()
