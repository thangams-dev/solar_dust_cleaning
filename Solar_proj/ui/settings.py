import streamlit as st

from core.safety import validate_servo_limits
from ui.components import section_header


def render(config, db, simulator, serial_manager):
    st.caption("Separate configuration panels keep device, safety, and storage options easy to find.")
    tabs = st.tabs([
        "Operating mode", "Hardware & sensor", "AI", "Cleaning",
        "Data storage", "Appearance",
    ])
    with st.form("settings-form"):
        with tabs[0]:
            section_header("Operating mode", "Demo Mode is the safe default when no hardware is attached.")
            mode = st.selectbox(
                "Operating mode", ["Demo Mode", "Hardware Mode"],
                index=0 if config.operating_mode == "Demo Mode" else 1,
            )
            st.caption("Hardware Mode uses actual serial telemetry only; it will not substitute demo readings.")
            serial_port = st.text_input(
                "Serial port", config.serial_port,
                placeholder="/dev/ttyUSB0 or COM3",
            )
            baud_rate = st.selectbox(
                "Baud rate", [9600, 19200, 38400, 57600, 115200],
                index=[9600, 19200, 38400, 57600, 115200].index(config.baud_rate)
                if config.baud_rate in [9600, 19200, 38400, 57600, 115200] else 4,
            )
        with tabs[1]:
            section_header("Hardware & sensor", "Laptop camera is accessed only when requested.")
            camera_index = st.number_input(
                "Webcam device index", min_value=0, max_value=20,
                value=config.camera_index, step=1,
            )
            ina219_address = st.text_input("INA219 I²C address", config.ina219_address)
            voltage_limit = st.number_input(
                "Maximum voltage (V)", min_value=0.1, value=float(config.voltage_limit)
            )
            current_limit = st.number_input(
                "Maximum current (A)", min_value=0.001, value=float(config.current_limit)
            )
            power_limit = st.number_input(
                "Maximum power (W)", min_value=0.01, value=float(config.power_limit)
            )
            sampling_interval = st.number_input(
                "Sampling interval (seconds)", min_value=1, max_value=60,
                value=int(config.sampling_interval),
            )
        with tabs[2]:
            section_header("AI configuration", "No trained model or real accuracy is bundled with this demo.")
            model_path = st.text_input(
                "Trained model path (optional)", st.session_state.get("model_path", "")
            )
            minimum_confidence = st.number_input(
                "Minimum confidence for cleaning (%)", min_value=0.0, max_value=100.0,
                value=float(config.minimum_confidence),
            )
            st.info("A model file alone is not sufficient; connect its inference loader before using real predictions.")
        with tabs[3]:
            section_header("Cleaning configuration", "Crack, shading, normal, or unknown states remain blocked.")
            auto_cleaning = st.checkbox(
                "Enable automatic cleaning when Dust is eligible",
                value=config.auto_cleaning,
            )
            dust_threshold = st.number_input(
                "Minimum dust coverage (%)", min_value=0.0, max_value=100.0,
                value=float(config.dust_threshold),
            )
            cleaning_timeout = st.number_input(
                "Cleaning timeout (seconds)", min_value=1, max_value=3600,
                value=int(config.cleaning_timeout),
            )
            servo_min_angle = st.number_input(
                "Servo minimum angle", min_value=0, max_value=179,
                value=int(config.servo_min_angle),
            )
            servo_max_angle = st.number_input(
                "Servo maximum angle", min_value=1, max_value=180,
                value=int(config.servo_max_angle),
            )
        with tabs[4]:
            section_header("Data storage", "Historical records are preserved when operating mode changes.")
            image_directory = st.text_input("Image storage folder", config.image_directory)
            database_path = st.text_input("SQLite database location", config.database_path)
        with tabs[5]:
            section_header("Appearance", "Choose a comfortable dashboard color scheme.")
            current_theme = st.session_state.get(
                "appearance_preference", "Dark" if st.session_state.get("dark_mode") else "Light"
            )
            appearance = st.selectbox(
                "Dashboard theme", ["Light", "Dark"],
                index=0 if current_theme == "Light" else 1,
            )
        submitted = st.form_submit_button("Save settings", type="primary")

    if submitted:
        try:
            old_mode = config.operating_mode
            active_cleaning = st.session_state.get("cleaning_status") in {
                "Cleaning in Progress", "Stop requested — awaiting ESP32"
            }
            if old_mode == "Hardware Mode" and active_cleaning:
                raise ValueError(
                    "Stop hardware cleaning and wait for ESP32 confirmation before changing settings."
                )
            validate_servo_limits(int(servo_min_angle), int(servo_max_angle))
            if voltage_limit <= 0 or current_limit <= 0 or power_limit <= 0:
                raise ValueError("Electrical limits must be greater than zero.")
            config.operating_mode = mode
            if old_mode != mode:
                st.session_state.last_inspection = None
                st.session_state.panel_condition = "Unknown"
                st.session_state.cleaning_status = "Idle"
            config.serial_port = serial_port.strip()
            config.baud_rate = int(baud_rate)
            config.camera_index = int(camera_index)
            config.ina219_address = ina219_address
            config.voltage_limit = float(voltage_limit)
            config.current_limit = float(current_limit)
            config.power_limit = float(power_limit)
            config.sampling_interval = int(sampling_interval)
            config.auto_cleaning = auto_cleaning
            st.session_state.auto_cleaning_control = auto_cleaning
            config.dust_threshold = float(dust_threshold)
            config.minimum_confidence = float(minimum_confidence)
            config.cleaning_timeout = int(cleaning_timeout)
            config.servo_min_angle = int(servo_min_angle)
            config.servo_max_angle = int(servo_max_angle)
            config.image_directory = image_directory
            if database_path != config.database_path:
                from database.db_manager import DatabaseManager
                config.database_path = database_path
                st.session_state.db = DatabaseManager(database_path)
            st.session_state.model_path = model_path
            st.session_state.appearance_preference = appearance
            if mode != "Hardware Mode":
                serial_manager.disconnect()
            st.success(f"Settings saved · {mode}")
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(f"Settings were not saved · {error}")

    if config.operating_mode == "Hardware Mode":
        section_header("Hardware connection")
        if st.button("Check ESP32 connection"):
            connected = serial_manager.connect(config.serial_port, config.baud_rate)
            st.session_state.serial_config_key = (
                config.operating_mode, config.serial_port, config.baud_rate
            )
            if connected:
                st.success(
                    f"Opened serial port {config.serial_port}; waiting for ESP32 telemetry."
                )
            else:
                st.error(f"ESP32 disconnected · {serial_manager.last_error}")

    with st.expander("Demo controls"):
        st.caption("Resetting demo state preserves all historical records.")
        if st.button("Reset Demo State"):
            simulator.set_scenario("Normal")
            st.session_state.demo_scenario = "Normal"
            st.session_state.demo_fault = ""
            st.session_state.panel_condition = (
                "Normal" if config.operating_mode == "Demo Mode" else "Unknown"
            )
            st.session_state.cleaning_status = "Idle"
            st.session_state.last_inspection = None
            st.session_state.cleaning_progress = 0
            st.session_state.last_cleaning_area = None
            st.session_state.cleaning_images = None
            st.success("Demo state reset; stored history was preserved.")
        st.caption(
            f"Database · `{config.database_path}` · Images · `{config.image_directory}`"
        )
        if st.checkbox("I understand this permanently deletes DEMO / SIMULATED records"):
            if st.button("Clear Demo Data", type="secondary"):
                db.clear_demo_data()
                st.success("Demo records deleted. Hardware records were preserved.")
