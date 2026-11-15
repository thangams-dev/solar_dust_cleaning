import streamlit as st

from core.safety import cleaning_decision, validate_servo_limits
from ui.cleaning import start_cleaning, stop_cleaning
from ui.components import local_timestamp, metric_cards, section_header, status_badge


WORKFLOW = (
    "Dust detection", "Safety verification", "Cleaning started",
    "Brush movement", "Cleaning completed", "Post-cleaning inspection",
)


def _workflow(condition: str, decision, status: str,
              post_inspection_done: bool) -> None:
    section_header("Cleaning workflow", "Safety checks must pass before the brush can move.")
    if "Completed" in status:
        states = ["Completed"] * 5 + [
            "Completed" if post_inspection_done else "Active"
        ]
    elif status == "Cleaning in Progress":
        states = ["Completed"] * 3 + ["Active", "Pending", "Pending"]
    elif "Failed" in status or "Stopped" in status:
        states = ["Completed", "Completed", "Completed", "Blocked", "Pending", "Pending"]
    elif condition == "Dust" and decision.allowed:
        states = ["Completed", "Completed"] + ["Pending"] * 4
    elif condition != "Unknown":
        states = ["Completed", "Blocked"] + ["Pending"] * 4
    else:
        states = ["Active"] + ["Pending"] * 5
    columns = st.columns(3)
    for index, (column, label) in enumerate(zip(columns * 2, WORKFLOW)):
        state = states[index]
        marker = {"Completed": "✓", "Active": "●", "Blocked": "×", "Pending": "○"}[state]
        with column:
            with st.container(border=True):
                st.markdown(f"**{marker}  {index + 1}. {label}**")
                st.caption("Safety block" if state == "Blocked" else state)


def render(db, config, serial_manager):
    status = st.session_state.get("cleaning_status", "Idle")
    inspection = st.session_state.get("last_inspection")
    expected_source = "Demo AI (simulated)" if config.operating_mode == "Demo Mode" else "Real AI"
    if not inspection or inspection.get("source") != expected_source:
        inspection = None
    condition = inspection["condition"] if inspection else "Unknown"
    confidence = float(inspection.get("confidence", 0)) if inspection else 0
    area = float(inspection.get("affected_area", 0)) if inspection else None
    decision = cleaning_decision(condition, confidence, config.minimum_confidence,
                                 area, config.dust_threshold)
    source = "Demo (simulated)" if config.operating_mode == "Demo Mode" else "Hardware"
    cleaning_history = db.records("cleanings", limit=100000, where="source = ?", params=(source,))
    last_cleaning = cleaning_history[0] if cleaning_history else {}
    servo_status = (
        "Simulated" if config.operating_mode == "Demo Mode"
        else st.session_state.get("hardware_status", {}).get("servo", "Unavailable")
    )
    metric_cards([
        {"label": "Panel condition", "value": condition, "icon": "◉",
         "note": f"AI source · {inspection['source']}" if inspection else "Inspection required"},
        {"label": "Cleaning mode", "value": "Auto" if config.auto_cleaning else "Manual",
         "icon": "↻", "note": "Auto mode never bypasses safety checks"},
        {"label": "Servo motor", "value": servo_status, "icon": "⚙",
         "note": "Status follows operating mode"},
        {"label": "Cleaning cycles", "value": str(len(cleaning_history)), "icon": "▤",
         "note": f"Last cleaning · {local_timestamp(last_cleaning.get('timestamp'))}"},
    ])
    with st.container(border=True):
        section_header("Eligibility", "Dust must meet the configured area and confidence thresholds.")
        if decision.allowed:
            st.success(decision.reason)
        else:
            st.warning(decision.reason)
        if inspection:
            st.caption(
                f"Inspection confidence · {confidence:.1f}% · "
                f"Estimated affected area · {area:.1f}% · {inspection['source']}"
            )
        try:
            validate_servo_limits(config.servo_min_angle, config.servo_max_angle)
            st.caption(
                f"Configured movement range · {config.servo_min_angle}°–"
                f"{config.servo_max_angle}° (validate the physical mechanism before use)"
            )
        except ValueError as error:
            st.error(str(error))

    controls = st.columns([1.2, 1, 1], vertical_alignment="center")
    if controls[0].button("START CLEANING", type="primary", width="stretch",
                          disabled=status == "Cleaning in Progress"):
        try:
            started = start_cleaning(
                db, config.operating_mode, condition, confidence,
                config.minimum_confidence, serial_manager,
                before_area=area, dust_threshold=config.dust_threshold,
                result_source=inspection.get("source") if inspection else None,
            )
            if started and config.operating_mode == "Hardware Mode":
                st.info("Command sent. Waiting for ESP32 confirmation.")
        except ValueError as error:
            st.error(f"Cleaning blocked · {error}")
    if controls[1].button("STOP CLEANING", width="stretch",
                          disabled=status not in {"Cleaning in Progress", "Stop requested — awaiting ESP32"}):
        stop_cleaning(db, config.operating_mode, serial_manager)
    if "auto_cleaning_control" not in st.session_state:
        st.session_state.auto_cleaning_control = config.auto_cleaning
    config.auto_cleaning = controls[2].toggle(
        "AUTO MODE", key="auto_cleaning_control",
        help="Automatic mode may start only after an eligible Dust inspection.",
    )

    status = st.session_state.get("cleaning_status", "Idle")
    with st.container(border=True):
        section_header("Cleaning progress", "Demo movement is simulated; hardware completion requires ESP32 confirmation.")
        status_badge(status)
        if status == "Cleaning in Progress":
            progress = st.session_state.get("cleaning_progress", 0)
            if config.operating_mode == "Demo Mode":
                st.progress(progress, text=f"DEMO / SIMULATED · Brush movement {progress}%")
            else:
                st.progress(0.0, text="Hardware cleaning in progress · awaiting ESP32 status")
        elif "Completed" in status:
            st.success(status)
        else:
            st.caption("No cleaning cycle is currently in progress.")

    _workflow(
        condition, decision, status,
        post_inspection_done=config.operating_mode == "Demo Mode",
    )
    if "Completed" in status and config.operating_mode == "Hardware Mode":
        st.caption("ESP32 confirmed the servo cycle; capture a follow-up panel image to complete inspection.")
    if st.session_state.get("last_cleaning_area"):
        before, after = st.session_state.last_cleaning_area
        section_header("Before and after", "The after-cleaning preview is simulated.")
        st.caption(f"Estimated dust coverage · {before:.1f}% → {after:.1f}%")
        if st.session_state.get("cleaning_images"):
            before_image, after_image = st.session_state.cleaning_images
            image_columns = st.columns(2)
            image_columns[0].image(
                before_image, caption="Before cleaning", width="stretch"
            )
            image_columns[1].image(
                after_image, caption="After cleaning · simulated", width="stretch"
            )

    section_header("Cleaning history")
    if cleaning_history:
        st.dataframe(cleaning_history, width="stretch", hide_index=True)
    else:
        st.caption("No cleaning cycles recorded for this operating mode.")
