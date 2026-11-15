import time

import streamlit as st

from core.safety import CleaningSafetyError, require_cleaning_allowed
from ui.common import add_alert, now_iso, simulated_cleaning_images


def start_cleaning(db, mode: str, condition: str, confidence: float,
                   threshold: float, serial_manager=None, reason: str = "Dust detected",
                   before_area: float | None = None, dust_threshold: float = 0,
                   result_source: str | None = None) -> bool:
    expected_source = "Demo AI (simulated)" if mode == "Demo Mode" else "Real AI"
    if result_source != expected_source:
        raise CleaningSafetyError(
            f"Cleaning requires a current {expected_source} inspection result."
        )
    require_cleaning_allowed(condition, confidence, threshold, before_area, dust_threshold)
    if mode == "Hardware Mode":
        if not serial_manager or not serial_manager.send("START_CLEANING"):
            message = serial_manager.last_error if serial_manager else "Serial connection is unavailable."
            add_alert(db, "Servo command failed", "Critical", message, "Hardware")
            st.error(f"ESP32 did not accept the cleaning command: {message}")
            return False
        st.session_state.cleaning_status = "Cleaning in Progress"
        st.session_state.cleaning_started = time.monotonic()
        st.session_state.hardware_cleaning_before = {
            "reason": reason, "condition": condition, "area": float(before_area or 0),
        }
        add_alert(db, "Cleaning started", "Info", "ESP32 accepted START_CLEANING; awaiting hardware confirmation.", "Hardware")
        return True

    st.session_state.cleaning_status = "Cleaning in Progress"
    st.session_state.cleaning_started = time.monotonic()
    st.session_state.cleaning_duration = 5.0
    st.session_state.cleaning_before = {
        "reason": reason, "condition": condition, "area": float(before_area or 0),
        "image": (st.session_state.get("inspection_image").copy()
                  if st.session_state.get("inspection_image") is not None else None),
    }
    st.session_state.cleaning_progress = 0
    return True


def stop_cleaning(db, mode: str, serial_manager=None) -> None:
    if mode == "Hardware Mode":
        if not serial_manager or not serial_manager.send("STOP_CLEANING"):
            message = serial_manager.last_error if serial_manager else "Serial connection is unavailable."
            add_alert(db, "Servo command failed", "Critical", message, "Hardware")
            st.error(f"ESP32 did not accept the stop command: {message}")
            return
        st.session_state.cleaning_status = "Stop requested — awaiting ESP32"
        return
    if st.session_state.get("cleaning_status") == "Cleaning in Progress":
        before = st.session_state.get("cleaning_before", {})
        db.insert("cleanings", {
            "timestamp": now_iso(), "reason": before.get("reason", "Dust detected"),
            "mode": "Demo Mode",
            "duration_seconds": round(time.monotonic() - st.session_state.cleaning_started, 2),
            "status": "Stopped (simulated)", "before_condition": before.get("condition", "Dust"),
            "after_condition": "Incomplete", "before_area": before.get("area", 0),
            "after_area": before.get("area", 0), "source": "Demo (simulated)",
        })
    st.session_state.cleaning_status = "Stopped (simulated)"
    st.session_state.cleaning_progress = 0
    add_alert(db, "Cleaning stopped", "Warning", "Demo cleaning was stopped by the operator.", "Demo (simulated)")


def advance_demo_cleaning(db) -> None:
    if st.session_state.get("cleaning_status") != "Cleaning in Progress":
        return
    duration = max(0.1, float(st.session_state.get("cleaning_duration", 5.0)))
    elapsed = max(0.0, time.monotonic() - st.session_state.cleaning_started)
    st.session_state.cleaning_progress = min(100, int(elapsed / duration * 100))
    if elapsed < duration:
        return
    before = st.session_state.get("cleaning_before", {})
    before_area = float(before.get("area", 0))
    after_area = round(before_area * 0.12, 1)
    db.insert("cleanings", {
        "timestamp": now_iso(), "reason": before.get("reason", "Dust detected"),
        "mode": "Demo Mode", "duration_seconds": round(elapsed, 2),
        "status": "Completed (simulated)",
        "before_condition": before.get("condition", "Dust"),
        "after_condition": "Normal (simulated)", "before_area": before_area,
        "after_area": after_area, "source": "Demo (simulated)",
    })
    add_alert(db, "Cleaning completed", "Info",
              "Simulated cleaning cycle completed; no physical cleaning occurred.",
              "Demo (simulated)")
    st.session_state.cleaning_status = "Cleaning Completed (simulated)"
    st.session_state.panel_condition = "Normal (simulated post-cleaning)"
    st.session_state.last_cleaning_area = (before_area, after_area)
    st.session_state.cleaning_images = simulated_cleaning_images(before.get("image"))
    st.session_state.last_inspection = None
    st.session_state.cleaning_progress = 100


def complete_hardware_cleaning(db, status: str) -> None:
    active_statuses = {"Cleaning in Progress", "Stop requested — awaiting ESP32"}
    if status == "COMPLETED" and st.session_state.get("cleaning_status") in active_statuses:
        db.insert("cleanings", {
            "timestamp": now_iso(), "reason": "Dust detected", "mode": "Hardware Mode",
            "duration_seconds": round(time.monotonic() - st.session_state.get("cleaning_started", time.monotonic()), 2),
            "status": "Completed (ESP32 confirmed)", "before_condition": st.session_state.get("panel_condition", "Dust"),
            "after_condition": "Awaiting post-cleaning inspection", "before_area": 0.0,
            "after_area": 0.0, "source": "Hardware",
        })
        st.session_state.cleaning_status = "Cleaning Completed (ESP32 confirmed)"
        st.session_state.last_inspection = None
        st.session_state.panel_condition = "Awaiting post-cleaning inspection"
        add_alert(db, "Cleaning completed", "Info", "ESP32 confirmed cleaning completion.", "Hardware")
    elif status in {"FAILED", "STOPPED"} and st.session_state.get("cleaning_status") in active_statuses:
        before = st.session_state.get("hardware_cleaning_before", {})
        db.insert("cleanings", {
            "timestamp": now_iso(), "reason": before.get("reason", "Dust detected"),
            "mode": "Hardware Mode",
            "duration_seconds": round(time.monotonic() - st.session_state.get("cleaning_started", time.monotonic()), 2),
            "status": status.title(), "before_condition": before.get("condition", "Unknown"),
            "after_condition": "Not completed", "before_area": before.get("area", 0),
            "after_area": before.get("area", 0), "source": "Hardware",
        })
        st.session_state.cleaning_status = status.title()
        add_alert(db, "Cleaning failed" if status == "FAILED" else "Cleaning stopped",
                  "Critical" if status == "FAILED" else "Warning",
                  f"ESP32 reported cleaning status: {status}.", "Hardware")


def check_hardware_cleaning_timeout(db, config, serial_manager) -> None:
    if st.session_state.get("cleaning_status") != "Cleaning in Progress":
        return
    started = st.session_state.get("cleaning_started")
    if started is None or time.monotonic() - started <= config.cleaning_timeout:
        return
    serial_manager.send("STOP_CLEANING")
    st.session_state.cleaning_status = "Cleaning Failed (timeout)"
    add_alert(db, "Cleaning failed", "Critical",
              "ESP32 did not confirm cleaning completion before the configured timeout.",
              "Hardware")
