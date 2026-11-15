from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_hardware_mode_disconnected_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SOLAR_DATABASE_PATH", str(tmp_path / "solar.db"))
    monkeypatch.setenv("SOLAR_IMAGE_DIRECTORY", str(tmp_path / "images"))
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    assert not app.exception

    app.radio(key="navigation").set_value("Settings").run()
    assert not app.exception
    app.selectbox[0].set_value("Hardware Mode")
    app.text_input[0].set_value("/dev/does-not-exist")
    app.form_submit_button[0].click().run()
    assert not app.exception
    for page in ("Overview", "Live Monitoring", "AI Inspection", "Automatic Cleaning",
                 "Historical Data", "Alerts & Logs", "System Status", "Settings"):
        app.radio(key="navigation").set_value(page).run()
        assert not app.exception
