from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image

from ai.inspection import InspectionService
from ai.model_manager import ModelManager
from hardware.camera_manager import CameraManager
from ui.cleaning import start_cleaning
from ui.common import add_alert, add_inspection
from ui.components import local_timestamp, section_header, status_badge


def _save_image(image: Image.Image, filename: str, folder: str) -> str:
    target_folder = Path(folder).expanduser().resolve()
    target_folder.mkdir(parents=True, exist_ok=True)
    target = target_folder / Path(filename).name
    image.convert("RGB").save(target, format="JPEG")
    return str(target)


def _show_result(db, config, serial_manager, image: Image.Image, filename: str) -> None:
    if config.operating_mode == "Demo Mode":
        condition = st.session_state.get("inspection_condition", "Dust")
        result = InspectionService().inspect_demo(image.copy(), condition, filename)
        st.info("Demo AI simulator only — not a trained model prediction or accuracy measurement.")
    else:
        model = ModelManager(st.session_state.get("model_path", ""))
        try:
            result = InspectionService(model).inspect_real(
                st.session_state.inspection_image_path
            )
        except (RuntimeError, ValueError, OSError) as error:
            st.error(f"Real AI inspection unavailable: {error}")
            return
    result["filename"] = filename
    add_inspection(db, result)
    st.session_state.last_inspection = result
    condition = result["condition"]
    st.session_state.panel_condition = condition
    st.success(f"{result['source']} result saved.")
    if condition == "Dust" and config.auto_cleaning:
        try:
            start_cleaning(
                db, config.operating_mode, condition, result["confidence"],
                config.minimum_confidence, serial_manager,
                before_area=result["affected_area"],
                dust_threshold=config.dust_threshold,
                result_source=result["source"],
            )
        except ValueError as error:
            st.info(f"Automatic cleaning not started: {error}")


def render(db, config, serial_manager):
    st.caption(
        "AI inference is a replaceable module. "
        + ("Demo AI results are simulated." if config.operating_mode == "Demo Mode"
           else "No trained model is loaded; inspection will not produce simulated hardware-mode results.")
    )
    section_header("Inspect a panel image", "Upload a photo or capture a frame from the laptop webcam.")
    left, right = st.columns([1.25, 0.9], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("**Image source**")
            upload = st.file_uploader("Upload panel image", type=["jpg", "jpeg", "png", "webp"])
            if upload:
                try:
                    image = Image.open(upload).convert("RGB")
                    st.session_state.inspection_image = image
                    st.session_state.inspection_filename = Path(upload.name).name
                except (OSError, ValueError) as error:
                    st.error(f"Could not open uploaded image: {error}")

            if st.session_state.get("camera_active"):
                try:
                    st.session_state.camera_frame = CameraManager.capture(config.camera_index)
                except RuntimeError as error:
                    st.session_state.camera_active = False
                    st.warning(f"{error} Upload an image to continue.")
                    if config.operating_mode == "Hardware Mode":
                        add_alert(db, "Webcam disconnected", "Warning", str(error), "Hardware")
            camera_controls = st.columns(3)
            if camera_controls[0].button("Start Camera", width="stretch"):
                try:
                    st.session_state.camera_frame = CameraManager.capture(config.camera_index)
                    st.session_state.camera_active = True
                    st.session_state.camera_last_capture = datetime.now().astimezone().isoformat()
                except RuntimeError as error:
                    st.session_state.camera_active = False
                    st.warning(f"{error} Upload an image to continue.")
                    if config.operating_mode == "Hardware Mode":
                        add_alert(db, "Webcam disconnected", "Warning", str(error), "Hardware")
            if camera_controls[1].button("Stop Camera", width="stretch"):
                st.session_state.camera_active = False
                st.session_state.camera_frame = None
            if camera_controls[2].button("Capture Image", width="stretch"):
                frame = st.session_state.get("camera_frame")
                if frame is None:
                    try:
                        frame = CameraManager.capture(config.camera_index)
                    except RuntimeError as error:
                        st.warning(f"{error} Upload an image to continue.")
                        if config.operating_mode == "Hardware Mode":
                            add_alert(db, "Webcam disconnected", "Warning", str(error), "Hardware")
                        frame = None
                if frame is not None:
                    st.session_state.inspection_image = frame.copy()
                    st.session_state.inspection_filename = "webcam-capture.jpg"
                    st.session_state.camera_last_capture = datetime.now().astimezone().isoformat()

            if st.session_state.get("camera_frame") is not None and st.session_state.get("camera_active"):
                st.image(st.session_state.camera_frame, caption="Webcam preview · on-demand frames",
                         width="stretch")
                st.caption("Camera refresh follows dashboard interval; it is not continuous video.")

            image = st.session_state.get("inspection_image")
            if image:
                st.image(image, caption=st.session_state.get("inspection_filename", "Panel image"),
                         width="stretch")
                if st.button("Clear selected image"):
                    st.session_state.inspection_image = None
                    st.session_state.inspection_image_path = None
                    st.rerun()
                if config.operating_mode == "Demo Mode":
                    st.selectbox("Demo inspection condition",
                                 ["Normal", "Dust", "Crack", "Partial Shading"],
                                 key="inspection_condition")
                else:
                    st.info("Hardware Mode requires a configured trained model. Demo AI is disabled.")
                label = "Run Demo AI Inspection" if config.operating_mode == "Demo Mode" else "Run Model Inspection"
                if st.button(label, type="primary", width="stretch"):
                    filename = st.session_state.get("inspection_filename", "uploaded-panel.jpg")
                    st.session_state.inspection_image_path = _save_image(
                        image, filename, config.image_directory
                    )
                    _show_result(db, config, serial_manager, image, filename)
            else:
                st.info("Upload a panel image or capture one from a connected webcam.")

    with right:
        with st.container(border=True):
            section_header("Inspection result", "Condition and recommended response")
            expected_source = (
                "Demo AI (simulated)" if config.operating_mode == "Demo Mode" else "Real AI"
            )
            latest = db.records("inspections", limit=1, where="source = ?",
                                params=(expected_source,))
            if latest:
                item = latest[0]
                label = item["condition"].upper()
                if item["source"].startswith("Demo"):
                    label += " · SIMULATED"
                status_badge(label)
                st.metric("Confidence", f"{item['confidence']:.1f}%")
                st.metric("Estimated affected area", f"{item['affected_area']:.1f}%")
                st.caption(f"Inspected · {local_timestamp(item['timestamp'])}")
                if item["condition"] == "Dust":
                    st.warning("Recommended action · Check cleaning eligibility.")
                elif item["condition"] == "Crack":
                    st.error("Critical · Do not clean. Inspect the panel for damage.")
                elif item["condition"] == "Partial Shading":
                    st.warning("Investigate the shading source. Cleaning is not permitted.")
                else:
                    st.success("Normal · No cleaning required.")
                if item["source"] == "Demo AI (simulated)":
                    st.caption("Confidence and affected area are simulated, not trained-model accuracy.")
                else:
                    st.caption("Model-reported confidence is not validated accuracy.")
            else:
                st.info("No inspection result for the selected operating mode.")

    if st.session_state.get("cleaning_images"):
        section_header("Before and after", "After-cleaning preview is simulated in Demo Mode.")
        before, after = st.session_state.cleaning_images
        image_columns = st.columns(2)
        area = st.session_state.get("last_cleaning_area", (0.0, 0.0))
        image_columns[0].image(
            before, caption=f"Before cleaning · {area[0]:.1f}% affected area",
            width="stretch",
        )
        image_columns[1].image(
            after, caption=f"After cleaning · {area[1]:.1f}% simulated estimate",
            width="stretch",
        )

    section_header("Inspection history")
    expected_source = (
        "Demo AI (simulated)" if config.operating_mode == "Demo Mode" else "Real AI"
    )
    history = db.records("inspections", limit=100, where="source = ?",
                         params=(expected_source,))
    if history:
        st.dataframe(history, width="stretch", hide_index=True)
