import json
from datetime import datetime


class ProtocolError(ValueError):
    pass


def parse_message(line: str) -> dict:
    try:
        message = json.loads(line)
    except (json.JSONDecodeError, TypeError) as error:
        raise ProtocolError("Serial message is not valid JSON.") from error
    if not isinstance(message, dict):
        raise ProtocolError("Serial JSON message must be an object.")
    message_type = message.get("type")
    if message_type == "sensor":
        try:
            voltage = float(message["voltage"])
            current = float(message["current"])
            power = float(message["power"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError("Sensor message requires numeric voltage, current, and power.") from error
        if not all(map(lambda value: value >= 0 and value < float("inf"),
                       (voltage, current, power))):
            raise ProtocolError("Sensor values must be finite and non-negative.")
        if abs(power - voltage * current) > max(0.05, voltage * current * 0.15):
            raise ProtocolError("Reported power is inconsistent with voltage × current.")
        timestamp = message.get("timestamp")
        if timestamp:
            try:
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError) as error:
                raise ProtocolError("Sensor timestamp must be ISO 8601.") from error
        return {"type": "sensor", "voltage": voltage, "current": current,
                "power": power, "timestamp": timestamp}
    if message_type == "cleaning_status" and message.get("status") in {
        "IDLE", "RUNNING", "COMPLETED", "FAILED", "STOPPED"
    }:
        return {"type": message_type, "status": message["status"]}
    if message_type == "error" and isinstance(message.get("message"), str):
        return {"type": "error", "message": message["message"]}
    raise ProtocolError("Unsupported or incomplete ESP32 message.")


def encode_command(command: str) -> bytes:
    if command not in {"START_CLEANING", "STOP_CLEANING"}:
        raise ProtocolError("Unsupported ESP32 command.")
    return (json.dumps({"command": command}, separators=(",", ":")) + "\n").encode()
