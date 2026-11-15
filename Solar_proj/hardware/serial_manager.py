from hardware.esp32_protocol import encode_command, parse_message


class SerialManager:
    def __init__(self) -> None:
        self.connection = None
        self.port = ""
        self.last_message = "No messages received"
        self.last_error = ""
        self.buffer = bytearray()

    @property
    def connected(self) -> bool:
        return bool(self.connection and self.connection.is_open)

    def connect(self, port: str, baud_rate: int) -> bool:
        self.disconnect()
        if not port:
            self.last_error = "No serial port configured."
            return False
        try:
            import serial
            self.connection = serial.Serial(port, baud_rate, timeout=0, write_timeout=1)
            self.port = port
            self.last_error = ""
            return True
        except (ImportError, OSError, ValueError) as error:
            self.last_error = str(error)
            self.connection = None
            return False

    def disconnect(self) -> None:
        if self.connection and self.connection.is_open:
            self.connection.close()
        self.connection = None
        self.buffer.clear()

    def send(self, command: str) -> bool:
        if not self.connected:
            self.last_error = "ESP32 is disconnected."
            return False
        try:
            self.connection.write(encode_command(command))
            return True
        except (OSError, ValueError) as error:
            self.last_error = str(error)
            return False

    def read_message(self) -> dict | None:
        if not self.connected:
            return None
        try:
            waiting = self.connection.in_waiting
            if waiting:
                self.buffer.extend(self.connection.read(waiting))
            if len(self.buffer) > 4096 and b"\n" not in self.buffer:
                self.buffer.clear()
                raise ValueError("Serial message exceeded the maximum line length.")
            if b"\n" not in self.buffer:
                return None
            raw_line, _, remainder = self.buffer.partition(b"\n")
            self.buffer = bytearray(remainder)
            if len(raw_line) > 2048:
                raise ValueError("Serial message exceeded the maximum line length.")
            line = raw_line.decode("utf-8", errors="strict").strip()
            if not line:
                return None
            message = parse_message(line)
            self.last_message = line
            return message
        except (OSError, UnicodeDecodeError, ValueError) as error:
            self.last_error = str(error)
            return {"type": "error", "message": self.last_error}
