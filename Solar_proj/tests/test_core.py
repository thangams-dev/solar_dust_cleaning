import random

import pytest
from PIL import Image

from ai.inspection import InspectionService
from core.safety import CleaningSafetyError, cleaning_decision, require_cleaning_allowed
from database.db_manager import DatabaseManager
from hardware.esp32_protocol import ProtocolError, encode_command, parse_message
from hardware.serial_manager import SerialManager
from simulation.ai_simulator import inspect
from simulation.sensor_simulator import SensorSimulator
from utils.calculations import accumulate_energy, calculate_power
from utils.exporters import records_to_csv


def test_power_and_elapsed_energy_account_for_changing_readings():
    assert calculate_power(6, 0.167) == pytest.approx(1.002)
    assert accumulate_energy(0, 1, 0.5, 3600) == pytest.approx(0.75)
    assert accumulate_energy(2, 0.4, 0.8, 1800) == pytest.approx(2.3)


def test_demo_sensor_generation_is_plausible_and_energy_never_decreases():
    simulator = SensorSimulator(seed=42)
    first = simulator.reading()
    second = simulator.reading()
    for reading in (first, second):
        assert 0.5 <= reading["voltage"] <= 6.0
        assert 0.005 <= reading["current"] <= 0.167
        assert reading["power"] == pytest.approx(reading["voltage"] * reading["current"])
    assert second["energy_wh"] >= first["energy_wh"]


def test_demo_ai_classifies_and_marks_simulation():
    result = inspect("Dust", "panel.jpg", random.Random(1))
    assert result["condition"] == "Dust"
    assert result["source"] == "Demo AI (simulated)"
    assert 0 <= result["affected_area"] <= 100
    image = Image.new("RGB", (4, 4))
    service_result = InspectionService().inspect_demo(image, "Normal", "panel.png")
    assert service_result["source"] == "Demo AI (simulated)"


@pytest.mark.parametrize("condition", ["Normal", "Crack", "Partial Shading", "Unknown"])
def test_cleaning_safety_blocks_non_dust_conditions(condition):
    assert not cleaning_decision(condition).allowed
    with pytest.raises(CleaningSafetyError):
        require_cleaning_allowed(condition)


def test_cleaning_allowed_only_for_confident_dust():
    assert cleaning_decision("Dust", confidence=80, threshold=60,
                             affected_area=30, dust_threshold=20).allowed
    assert not cleaning_decision("Dust", confidence=59, threshold=60,
                                 affected_area=30, dust_threshold=20).allowed
    assert not cleaning_decision("Dust", confidence=80, threshold=60,
                                 affected_area=10, dust_threshold=20).allowed


def test_serial_protocol_validates_sensor_and_commands():
    data = parse_message('{"type":"sensor","voltage":5.8,"current":0.145,"power":0.841}')
    assert data["power"] == pytest.approx(0.841)
    assert encode_command("START_CLEANING") == b'{"command":"START_CLEANING"}\n'
    assert parse_message('{"type":"cleaning_status","status":"COMPLETED"}')["status"] == "COMPLETED"


def test_serial_manager_buffers_partial_newline_delimited_messages():
    class FakeConnection:
        is_open = True

        def __init__(self):
            self.pending = b""

        @property
        def in_waiting(self):
            return len(self.pending)

        def read(self, count):
            result, self.pending = self.pending[:count], self.pending[count:]
            return result

    manager = SerialManager()
    manager.connection = FakeConnection()
    manager.connection.pending = b'{"type":"sensor","voltage":5.8,'
    assert manager.read_message() is None
    manager.connection.pending = b'"current":0.145,"power":0.841}\n'
    assert manager.read_message()["type"] == "sensor"


@pytest.mark.parametrize("message", [
    "not-json",
    "[]",
    '{"type":"sensor","voltage":-1,"current":0.1,"power":0.1}',
    '{"type":"sensor","voltage":5,"current":0.1,"power":4}',
    '{"type":"cleaning_status","status":"UNKNOWN"}',
])
def test_invalid_serial_messages_are_rejected(message):
    with pytest.raises(ProtocolError):
        parse_message(message)


def test_database_storage_retrieval_and_csv_export(tmp_path):
    db = DatabaseManager(str(tmp_path / "solar.db"))
    db.insert("readings", {"timestamp": "2026-01-01T10:00:00+00:00",
                           "voltage": 5.5, "current": 0.1, "power": 0.55,
                           "energy_wh": 0.01, "source": "Demo (simulated)"})
    rows = db.records("readings")
    assert len(rows) == 1
    assert rows[0]["power"] == pytest.approx(0.55)
    csv = records_to_csv(rows)
    assert "timestamp,voltage,current,power,energy_wh,source" in csv
    assert "Demo (simulated)" in csv
    db.clear_demo_data()
    assert db.records("readings") == []
