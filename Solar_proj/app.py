import time
from datetime import datetime, timezone

import streamlit as st

from core.config import AppConfig, ensure_directories
from database.db_manager import DatabaseManager
from hardware.serial_manager import SerialManager
from simulation.sensor_simulator import SensorSimulator
from ui import alerts_logs, ai_inspection, automatic_cleaning, historical_data
from ui import live_monitoring, overview, settings, system_status
from ui.cleaning import advance_demo_cleaning, check_hardware_cleaning_timeout
from ui.cleaning import complete_hardware_cleaning
from ui.common import add_alert, now_iso
from ui.components import status_badge
from ui.theme import dashboard_css
from utils.calculations import accumulate_energy


PAGES = {
    "Overview": "⌂",
    "Live Monitoring": "◉",
    "AI Inspection": "◎",
    "Automatic Cleaning": "↻",
    "Historical Data": "▤",
    "Alerts & Logs": "⚑",
    "System Status": "◇",
    "Settings": "⚙",
}

PAGE_META = {
    "Overview": ("Solar operations overview", "Live energy generation, panel condition, cleaning readiness, and device health."),
    "Live Monitoring": ("Live monitoring", "Electrical measurements and trends from the selected data source."),
    "AI Inspection": ("AI panel inspection", "Inspect an uploaded or captured panel image and review the reported condition."),
    "Automatic Cleaning": ("Automatic cleaning", "Review cleaning eligibility and control the safety-checked cleaning workflow."),
    "Historical Data": ("Historical data", "Explore saved readings, inspections, and cleaning records."),
    "Alerts & Logs": ("Alerts & logs", "Review system events, acknowledge alerts, and archive resolved items."),
    "System Status": ("System status", "Check current device connections and application health."),
    "Settings": ("Settings", "Configure operating mode, devices, inspection, cleaning, storage, and appearance."),
}


def _initialize() -> None:
    if "config" not in st.session_state:
        st.session_state.config = AppConfig()
        ensure_directories(st.session_state.config)
        st.session_state.db = DatabaseManager(st.session_state.config.database_path)
        previous_demo = st.session_state.db.records(
            "readings", limit=1, where="source LIKE 'Demo%'"
        )
        previous_hardware = st.session_state.db.records(
            "readings", limit=1, where="source = 'Hardware'"
        )
        demo_energy = float(previous_demo[0]["energy_wh"]) if previous_demo else 0.0
        hardware_energy = float(previous_hardware[0]["energy_wh"]) if previous_hardware else 0.0
        st.session_state.simulator = SensorSimulator(initial_energy_wh=demo_energy)
        st.session_state.hardware_energy_wh = hardware_energy
        st.session_state.previous_hardware_reading = None
        st.session_state.serial_manager = SerialManager()
        st.session_state.started_at = time.monotonic()
        latest_inspections = st.session_state.db.records(
            "inspections", limit=1, where="source = 'Demo AI (simulated)'"
        )
        latest_cleanings = st.session_state.db.records(
            "cleanings", limit=1, where="source = 'Demo (simulated)'"
        )
        st.session_state.last_inspection = latest_inspections[0] if latest_inspections else None
        st.session_state.panel_condition = (
            latest_inspections[0]["condition"] if latest_inspections else "Normal"
        )
        st.session_state.cleaning_status = "Idle"
        if latest_cleanings:
            latest_cleaning = latest_cleanings[0]
            cleaning_is_current = (
                not latest_inspections
                or latest_cleaning["timestamp"] >= latest_inspections[0]["timestamp"]
            )
            if cleaning_is_current:
                st.session_state.cleaning_status = latest_cleaning["status"]
            if cleaning_is_current and "Completed" in latest_cleaning["status"]:
                st.session_state.panel_condition = latest_cleaning["after_condition"]
                st.session_state.last_cleaning_area = (
                    latest_cleaning["before_area"], latest_cleaning["after_area"]
                )
        st.session_state.demo_scenario = "Normal"
        st.session_state.hardware_status = {}
        st.session_state.dark_mode = False


def _page_header(name: str, config: AppConfig, serial_manager: SerialManager) -> None:
    title, description = PAGE_META[name]
    with st.container():
        left, right = st.columns([1.5, 1], vertical_alignment="center")
        with left:
            st.markdown(
                f'<div class="top-header-copy"><h1>{title}</h1>'
                f'<p>{description}</p></div>',
                unsafe_allow_html=True,
            )
        with right:
            mode = "DEMO MODE" if config.operating_mode == "Demo Mode" else "HARDWARE MODE"
            connection = (
                "SIMULATED DEVICES" if config.operating_mode == "Demo Mode"
                else ("SERIAL LINK OPEN" if serial_manager.connected else "SERIAL LINK CLOSED")
            )
            badges = st.columns(2, gap="small")
            with badges[0]:
                status_badge(mode)
            with badges[1]:
                status_badge(connection)
            meta = st.columns([1, 1], vertical_alignment="center")
            meta[0].caption(datetime.now().astimezone().strftime("%b %d · %H:%M:%S"))
            meta[1].toggle("Dark mode", key="dark_mode")


def _store_reading(data: dict, source: str) -> None:
    st.session_state.db.insert("readings", {
        "timestamp": data.get("timestamp") or now_iso(),
        "voltage": float(data["voltage"]),
        "current": float(data["current"]),
        "power": float(data["power"]),
        "energy_wh": float(data["energy_wh"]),
        "source": source,
    })


def _update_hardware() -> None:
    config = st.session_state.config
    manager = st.session_state.serial_manager
    connection_key = (config.operating_mode, config.serial_port, config.baud_rate)
    if connection_key != st.session_state.get("serial_config_key"):
        manager.disconnect()
        st.session_state.serial_config_key = connection_key
        if config.operating_mode == "Hardware Mode":
            manager.connect(config.serial_port, config.baud_rate)
    if config.operating_mode != "Hardware Mode":
        manager.disconnect()
        st.session_state.hardware_status = {"esp32": "Simulated", "sensor": "Simulated"}
        return

    st.session_state.hardware_status = {
        "esp32": "Connected" if manager.connected else "Disconnected",
        "sensor": "Awaiting data" if manager.connected else "Disconnected",
        "servo": "Awaiting ESP32 status" if manager.connected else "Unavailable",
    }
    was_connected = st.session_state.get("last_esp32_connected")
    if manager.connected != was_connected:
        status = "Connected" if manager.connected else "Disconnected"
        add_alert(st.session_state.db, "ESP32 connection changed",
                  "Info" if manager.connected else "Warning",
                  f"ESP32 is now {status.lower()}.", "Hardware")
        st.session_state.last_esp32_connected = manager.connected
        st.session_state.hardware_connected_at = time.monotonic()
        st.session_state.sensor_disconnect_alerted = False
    message = manager.read_message()
    if not message:
        last_sensor = st.session_state.get("last_sensor_monotonic")
        connection_age = time.monotonic() - st.session_state.get("hardware_connected_at", time.monotonic())
        sensor_age = time.monotonic() - last_sensor if last_sensor else connection_age
        if manager.connected and sensor_age > 10:
            st.session_state.hardware_status["sensor"] = "No sensor data"
            if not st.session_state.get("sensor_disconnect_alerted"):
                add_alert(st.session_state.db, "INA219 disconnected", "Warning",
                          "No valid INA219 sensor message was received for 10 seconds.",
                          "Hardware")
                st.session_state.sensor_disconnect_alerted = True
        return
    if message["type"] == "sensor":
        voltage, current, power = message["voltage"], message["current"], message["power"]
        if (voltage > config.voltage_limit or current > config.current_limit
                or power > config.power_limit):
            add_alert(st.session_state.db, "Invalid sensor data", "Warning",
                      "ESP32 sensor reading exceeded a configured electrical limit; reading was not stored.",
                      "Hardware")
            return
        previous = st.session_state.get("previous_hardware_reading")
        energy = float(previous["energy_wh"]) if previous else st.session_state.hardware_energy_wh
        timestamp = datetime.now(timezone.utc)
        if previous:
            elapsed = max(0.0, (timestamp - previous["timestamp"]).total_seconds())
            energy = accumulate_energy(energy, previous["power"], power, elapsed)
        st.session_state.previous_hardware_reading = {
            "timestamp": timestamp, "power": power, "energy_wh": energy,
        }
        st.session_state.hardware_energy_wh = energy
        data = {"voltage": voltage, "current": current, "power": power,
                "energy_wh": energy, "timestamp": message.get("timestamp") or timestamp.isoformat()}
        _store_reading(data, "Hardware")
        st.session_state.hardware_status["sensor"] = "Receiving data"
        st.session_state.last_sensor_update = timestamp.isoformat()
        st.session_state.last_sensor_monotonic = time.monotonic()
        st.session_state.sensor_disconnect_alerted = False
    elif message["type"] == "cleaning_status":
        st.session_state.hardware_status["servo"] = message["status"]
        complete_hardware_cleaning(st.session_state.db, message["status"])
    elif message["type"] == "error":
        add_alert(st.session_state.db, "Invalid sensor data", "Warning",
                  message["message"], "Hardware")
        st.session_state.hardware_status["last_error"] = message["message"]


def _update_demo() -> None:
    fault = st.session_state.get("demo_fault", "")
    now = time.monotonic()
    last_sample = st.session_state.get("last_demo_sample_time")
    if (fault != "Sensor Disconnected"
            and (last_sample is None or now - last_sample >= st.session_state.config.sampling_interval)):
        data = st.session_state.simulator.reading()
        data["timestamp"] = now_iso()
        _store_reading(data, "Demo (simulated)")
        st.session_state.last_demo_sample_time = now
    advance_demo_cleaning(st.session_state.db)
    st.session_state.hardware_status = {
        "esp32": "Disconnected (demo scenario)" if fault == "ESP32 Disconnected" else "Simulated",
        "sensor": "Disconnected (demo scenario)" if fault == "Sensor Disconnected" else "Simulated",
        "webcam": "Available on request",
        "servo": "Simulated",
    }


def _render_page(name: str) -> None:
    config = st.session_state.config
    db = st.session_state.db
    simulator = st.session_state.simulator
    serial_manager = st.session_state.serial_manager
    _page_header(name, config, serial_manager)
    if config.operating_mode == "Demo Mode":
        _update_demo()
    else:
        _update_hardware()
        check_hardware_cleaning_timeout(db, config, serial_manager)
    page_handlers = {
        "Overview": lambda: overview.render(db, config, simulator, serial_manager),
        "Live Monitoring": lambda: live_monitoring.render(db, config, serial_manager),
        "AI Inspection": lambda: ai_inspection.render(db, config, serial_manager),
        "Automatic Cleaning": lambda: automatic_cleaning.render(db, config, serial_manager),
        "Historical Data": lambda: historical_data.render(db, config),
        "Alerts & Logs": lambda: alerts_logs.render(db),
        "System Status": lambda: system_status.render(db, config, serial_manager),
        "Settings": lambda: settings.render(config, db, simulator, serial_manager),
    }
    page_handlers[name]()


def main() -> None:
    st.set_page_config(page_title="Solar Insight | Panel Monitor",
                       page_icon="☀", layout="wide", initial_sidebar_state="expanded")
    _initialize()
    if "appearance_preference" in st.session_state:
        st.session_state.dark_mode = st.session_state.appearance_preference == "Dark"
    st.markdown(dashboard_css(st.session_state.dark_mode), unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark"><div class="brand-icon">☀</div><div>'
            '<div class="brand-name">SOLAR AI</div>'
            '<div class="brand-subtitle">Solar Monitoring System</div></div></div>',
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", list(PAGES),
                        format_func=lambda value: f"{PAGES[value]}  {value}",
                        key="navigation")
        st.divider()
        config = st.session_state.config
        st.caption("OPERATING MODE")
        status_badge("DEMO MODE" if config.operating_mode == "Demo Mode" else "HARDWARE MODE")
        connection = (
            "SIMULATED DEVICES" if config.operating_mode == "Demo Mode"
            else ("SERIAL LINK OPEN" if st.session_state.serial_manager.connected
                  else "SERIAL LINK CLOSED")
        )
        status_badge(connection)
        st.caption(f"Refresh interval · {config.sampling_interval}s")
        if st.button("Reset Demo State", width="stretch"):
            st.session_state.simulator.set_scenario("Normal")
            st.session_state.panel_condition = "Normal" if config.operating_mode == "Demo Mode" else "Unknown"
            st.session_state.cleaning_status = "Idle"
            st.session_state.cleaning_progress = 0
            st.session_state.demo_scenario = "Normal"
            st.session_state.demo_fault = ""
            st.session_state.last_inspection = None
            st.session_state.last_cleaning_area = None
            st.session_state.cleaning_images = None
            st.success("Demo state reset; history was preserved.")
    run_page = st.fragment(run_every=1)(
        lambda: _render_page(page)
    )
    run_page()


if __name__ == "__main__":
    main()
