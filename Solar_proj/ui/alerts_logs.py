import pandas as pd
import streamlit as st

from ui.components import local_timestamp, section_header, status_badge


def _component(alert_type: str) -> str:
    lowered = alert_type.lower()
    if "ina219" in lowered or "sensor" in lowered:
        return "Electrical sensor"
    if "esp32" in lowered or "serial" in lowered:
        return "ESP32"
    if "webcam" in lowered or "camera" in lowered:
        return "USB webcam"
    if "servo" in lowered or "cleaning" in lowered:
        return "Cleaning system"
    if "crack" in lowered or "dust" in lowered or "shading" in lowered:
        return "AI inspection"
    return "System"


def render(db):
    view = st.radio(
        "Alert view", ["All Alerts", "Active Alerts", "Critical Alerts", "Archived"],
        horizontal=True,
    )
    filters = {
        "All Alerts": ("archived = 0", ()),
        "Active Alerts": ("acknowledged = 0 AND archived = 0", ()),
        "Critical Alerts": ("severity = ? AND archived = 0", ("Critical",)),
        "Archived": ("archived = 1", ()),
    }
    where, params = filters[view]
    alerts = db.records("alerts", limit=1000, where=where, params=params)
    section_header("System alerts", f"{len(alerts)} items · {view.lower()}")
    if not alerts:
        st.success("No alerts in this view.")
        return

    frame = pd.DataFrame(alerts)
    frame["component"] = frame["alert_type"].map(_component)
    frame["timestamp"] = frame["timestamp"].map(local_timestamp)
    frame["acknowledgement"] = frame["acknowledged"].map(
        lambda value: "Acknowledged" if value else "Unacknowledged"
    )
    st.dataframe(
        frame[["severity", "alert_type", "description", "component",
               "timestamp", "source", "acknowledgement"]],
        width="stretch", hide_index=True,
    )
    section_header("Alert actions", "Acknowledge reviewed alerts or archive resolved ones.")
    for item in alerts:
        with st.container(border=True):
            left, middle, right = st.columns([0.19, 0.57, 0.24], vertical_alignment="center")
            left.markdown(f"**{item['alert_type']}**")
            left.caption(_component(item["alert_type"]))
            middle.write(item["description"])
            middle.caption(
                f"{local_timestamp(item['timestamp'])} · {item['source']} · "
                f"{'Acknowledged' if item['acknowledged'] else 'Needs review'}"
            )
            with right:
                status_badge(item["severity"])
                if not item["acknowledged"] and st.button(
                    "Acknowledge", key=f"ack-{item['id']}", width="stretch"
                ):
                    db.update_alert(item["id"], acknowledged=1)
                    st.rerun()
                if not item["archived"] and st.button(
                    "Archive", key=f"archive-{item['id']}", width="stretch"
                ):
                    db.update_alert(item["id"], archived=1)
                    st.rerun()
