import streamlit as st

from simulation.ai_simulator import CONDITIONS, inspect
from ui.cleaning import start_cleaning
from ui.common import add_alert, add_inspection
from ui.components import (
    activity_timeline,
    electrical_figure,
    filter_readings,
    local_timestamp,
    metric_cards,
    section_header,
    status_badge,
)


def _trend(current: dict, previous: dict | None, key: str, unit: str) -> str:
    if not current or not previous:
        return "Waiting for a comparison reading"
    difference = float(current[key]) - float(previous[key])
    if key == "current":
        difference *= 1000
    return f"{difference:+.3f} {unit} since previous reading"


def _recent_activity(db) -> list[dict]:
    events = []
    for table in ("inspections", "cleanings", "alerts", "readings"):
        records = db.records(table, limit=5)
        for record in records:
            item = dict(record)
            if table == "inspections":
                item["alert_type"] = f"AI inspection · {item['condition']}"
            elif table == "cleanings":
                item["alert_type"] = f"Cleaning · {item['status']}"
            elif table == "readings":
                item["alert_type"] = "Electrical reading recorded"
                item["description"] = (
                    f"{item['power']:.3f} W · {item['source']}"
                )
            events.append(item)
    return events


def _run_demo_scenario(db, config, simulator, serial_manager, scenario: str) -> None:
    st.session_state.demo_scenario = scenario
    if scenario in CONDITIONS:
        st.session_state.demo_fault = ""
        simulator.set_scenario(scenario)
        result = inspect(scenario)
        result["source"] = "Demo AI (simulated)"
        result["filename"] = "demo-scenario.jpg"
        add_inspection(db, result)
        st.session_state.panel_condition = scenario
        st.session_state.last_inspection = result
        if scenario == "Dust" and config.auto_cleaning:
            try:
                start_cleaning(
                    db, config.operating_mode, scenario, result["confidence"],
                    config.minimum_confidence, serial_manager,
                    before_area=result["affected_area"],
                    dust_threshold=config.dust_threshold,
                    result_source=result["source"],
                )
            except ValueError as error:
                st.warning(f"Automatic cleaning not started: {error}")
    elif scenario == "Low Solar Generation":
        st.session_state.demo_fault = ""
        simulator.set_scenario("Normal", low_generation=True)
    elif scenario == "Cleaning Successful":
        st.session_state.demo_fault = ""
        simulator.set_scenario("Dust")
        result = inspect("Dust")
        result["source"] = "Demo AI (simulated)"
        result["filename"] = "demo-cleaning.jpg"
        add_inspection(db, result)
        st.session_state.panel_condition = "Dust"
        st.session_state.last_inspection = result
        start_cleaning(
            db, "Demo Mode", "Dust", result["confidence"],
            config.minimum_confidence, before_area=result["affected_area"],
            dust_threshold=config.dust_threshold,
            result_source=result["source"],
        )
    else:
        st.session_state.demo_fault = scenario
        if scenario == "Sensor Disconnected":
            add_alert(
                db, "INA219 disconnected", "Warning",
                "DEMO scenario: sensor readings are paused to simulate a disconnected sensor.",
                "Demo (simulated)",
            )
        elif scenario == "ESP32 Disconnected":
            add_alert(
                db, "ESP32 disconnected", "Warning",
                "DEMO scenario: ESP32 is shown as disconnected; no actual hardware failure occurred.",
                "Demo (simulated)",
            )


def render(db, config, simulator, serial_manager):
    source = "Demo (simulated)" if config.operating_mode == "Demo Mode" else "Hardware"
    inspection_source = "Demo AI (simulated)" if config.operating_mode == "Demo Mode" else "Real AI"
    reading_records = db.records("readings", limit=2, where="source = ?", params=(source,))
    reading = reading_records[0] if reading_records else {}
    previous = reading_records[1] if len(reading_records) > 1 else None
    if config.operating_mode == "Demo Mode" and st.session_state.get("demo_fault") == "Sensor Disconnected":
        reading = {}
        reading_records = []
    inspections = db.records("inspections", limit=1, where="source = ?", params=(inspection_source,))
    cleanings = db.records("cleanings", limit=1, where="source = ?", params=(source,))
    inspection = inspections[0] if inspections else {}
    cleaning = cleanings[0] if cleanings else {}

    mode_note = "DEMO / SIMULATED DATA" if config.operating_mode == "Demo Mode" else "HARDWARE TELEMETRY"
    st.caption(mode_note)
    metric_cards([
        {
            "label": "Voltage", "icon": "ϟ",
            "value": f"{reading['voltage']:.2f}" if reading else "—",
            "unit": "V",
            "delta": _trend(reading, previous, "voltage", "V"),
        },
        {
            "label": "Current", "icon": "↗",
            "value": f"{reading['current'] * 1000:.1f}" if reading else "—",
            "unit": "mA",
            "delta": _trend(reading, previous, "current", "mA"),
        },
        {
            "label": "Power", "icon": "▰",
            "value": f"{reading['power']:.3f}" if reading else "—",
            "unit": "W", "note": "Panel rating · 1 W",
        },
        {
            "label": "Energy generated", "icon": "☀",
            "value": f"{reading['energy_wh']:.4f}" if reading else "—",
            "unit": "Wh", "note": "Integrated using elapsed time",
        },
    ])

    condition = st.session_state.get("panel_condition", "Unknown")
    cleaning_completed = (
        "Completed" in st.session_state.get("cleaning_status", "")
        and bool(cleaning)
        and (not inspection or cleaning["timestamp"] >= inspection["timestamp"])
    )
    if cleaning_completed:
        condition = cleaning["after_condition"]
    elif inspection:
        condition = inspection["condition"]
    cleaned_area = (
        float(cleaning.get("after_area", 0))
        if cleaning_completed and cleaning else
        float(inspection.get("affected_area", 0)) if inspection else None
    )
    inspection_columns = st.columns([1.1, 0.9], gap="medium")
    with inspection_columns[0]:
        with st.container(border=True):
            section_header("Panel condition", "Latest inspection and recommendation")
            status_badge(condition)
            if inspection:
                if cleaned_area is not None:
                    st.markdown(f"**Estimated affected area:** {cleaned_area:.1f}%")
                st.caption(f"Last inspection · {local_timestamp(inspection['timestamp'])}")
                st.caption(f"Detection source · {inspection['source']}")
                if cleaning_completed and config.operating_mode == "Demo Mode":
                    st.success("Simulated cleaning completed. No physical panel cleaning occurred.")
                elif cleaning_completed:
                    st.info("ESP32 confirmed the cycle. Run a follow-up inspection to verify the panel.")
                elif inspection["condition"] == "Dust":
                    st.warning("Dust detected. Review the cleaning safety check.")
                elif inspection["condition"] == "Crack":
                    st.error("Possible panel damage. Cleaning is prohibited.")
                elif inspection["condition"] == "Partial Shading":
                    st.warning("Partial shading detected. Cleaning is prohibited.")
                else:
                    st.success("No cleaning action is required.")
            else:
                st.caption("No inspection result for the current operating mode yet.")
    with inspection_columns[1]:
        with st.container(border=True):
            section_header("Cleaning status", "Automatic cleaning workflow")
            status_badge(st.session_state.get("cleaning_status", "Idle"))
            st.write(f"**Mode:** {'Automatic' if config.auto_cleaning else 'Manual'}")
            servo = (
                "Simulated" if config.operating_mode == "Demo Mode"
                else st.session_state.get("hardware_status", {}).get("servo", "Unavailable")
            )
            st.write(f"**Servo:** {servo}")
            st.write(f"**Last cleaning:** {local_timestamp(cleaning.get('timestamp'))}")
            if st.session_state.get("cleaning_status") == "Cleaning in Progress":
                amount = st.session_state.get("cleaning_progress", 0)
                st.progress(amount, text=f"DEMO / SIMULATED · Brush movement {amount}%")
            if condition != "Dust":
                st.caption("Cleaning is allowed only after an eligible Dust inspection.")

    section_header("Energy generation", "Voltage, current, and power trends · units shown on each axis")
    graph_controls = st.columns([0.38, 0.62], vertical_alignment="center")
    period = graph_controls[0].selectbox(
        "Time range", ["1 Hour", "6 Hours", "Today", "7 Days"],
        key="overview_range", label_visibility="collapsed",
    )
    graph_controls[1].caption(f"Data source · {source}")
    readings = db.records("readings", limit=10000, where="source = ?", params=(source,), descending=False)
    if config.operating_mode == "Demo Mode" and st.session_state.get("demo_fault") == "Sensor Disconnected":
        readings = []
    filtered = filter_readings(readings, period)
    with st.container(border=True):
        figure = electrical_figure(filtered)
        if figure is None:
            st.info("No readings in this time range. Check the sensor connection or choose a wider period.")
        else:
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    section_header("Device status", "Simulated devices are not physical connections")
    hardware_status = st.session_state.get("hardware_status", {})
    webcam_status = (
        "Not checked" if config.operating_mode == "Hardware Mode" else "Simulated / available on request"
    )
    ai_status = (
        "Not Configured" if config.operating_mode == "Hardware Mode"
        else "Demo AI · simulated"
    )
    device_items = [
        ("ESP32", hardware_status.get("esp32", "Simulated") if config.operating_mode == "Demo Mode"
         else ("Serial port open · device unverified" if serial_manager.connected else "Disconnected")),
        ("INA219", hardware_status.get("sensor", "Simulated") if config.operating_mode == "Demo Mode"
         else hardware_status.get("sensor", "Unavailable")),
        ("USB webcam", webcam_status),
        ("AI engine", ai_status),
        ("Servo motor", "Simulated" if config.operating_mode == "Demo Mode"
         else hardware_status.get("servo", "Unavailable")),
    ]
    device_columns = st.columns(5, gap="small")
    for column, (name, value) in zip(device_columns, device_items):
        with column:
            with st.container(border=True):
                st.caption(name)
                status_badge(value)

    activity = _recent_activity(db)
    with st.container(border=True):
        section_header("Recent activity", "Latest system events")
        activity_timeline(activity)

    if config.operating_mode == "Demo Mode":
        with st.expander("Demo scenarios · simulated only"):
            st.caption("Creates DEMO / SIMULATED records only. No physical panel or motor is controlled.")
            scenarios = (*CONDITIONS, "Low Solar Generation", "Cleaning Successful",
                         "Sensor Disconnected", "ESP32 Disconnected")
            scenario_columns = st.columns(4)
            for column, scenario in zip(scenario_columns * 2, scenarios):
                if column.button(scenario, width="stretch", key=f"scenario-{scenario}"):
                    _run_demo_scenario(db, config, simulator, serial_manager, scenario)
                    st.rerun()
            st.caption(f"Active scenario · {st.session_state.get('demo_scenario', 'Normal')}")
