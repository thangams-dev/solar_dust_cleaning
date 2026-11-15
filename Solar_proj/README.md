# Solar Insight — AI-Based Solar Panel Monitoring & Cleaning

A local Streamlit dashboard for demonstrating electrical monitoring, panel
inspection, alerts, and a safety-gated cleaning workflow. Demo Mode is the
default and labels simulated readings and AI results. Hardware Mode uses only
ESP32 serial telemetry and commands; disconnected hardware is reported as
disconnected, never replaced with simulated readings.

## Working principle

1. Voltage/current readings are recorded with their source; power is calculated
   as V × A and energy is integrated with elapsed time using trapezoidal power.
2. The AI interface accepts an optional future trained model. Without one, the
   panel inspection uses an explicitly labelled Demo AI simulator, not a claim
   of trained-model accuracy.
3. Cleaning safety is enforced in Python: only a sufficiently confident Dust
   result is eligible. Normal, Crack, Partial Shading, and unknown conditions
   cannot initiate cleaning.
4. Demo cleaning is simulated and never claims physical cleaning. Hardware
   cleaning is only marked completed after ESP32 confirmation.
5. SQLite stores readings, inspections, cleaning cycles, alerts, and data
   sources persistently across mode changes.

## Dashboard appearance

The interface uses a navy industrial-style sidebar, a light analytics workspace,
consistent status badges, mode-aware page headers, interactive charts, and an
optional dark theme. The Overview emphasizes electrical readings, panel
condition, cleaning readiness, and device status; the remaining pages use the
same visual system for inspection, controls, history, alerts, and settings.

## Architecture

```text
app.py                         Streamlit shell, navigation, mode-aware updates
ui/                            Dashboard page renderers and UI workflows
core/                          Configuration and cleaning safety
simulation/                    Simulated sensors and AI results
ai/                            Replaceable model/inference integration interface
hardware/                      Camera and newline-delimited JSON serial transport
database/                      SQLite schema and persistence
utils/                         Power/energy calculations and CSV export
esp32_firmware/                 Protocol example sketch
tests/                         Unit and integration tests
data/                           Local database and uploaded images (created at run)
```

## Requirements and installation

Python 3.10+ is recommended. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Debian/Ubuntu, if virtual-environment creation reports that `ensurepip` is
missing, install the matching `python3-venv` package first (for example,
`sudo apt install python3.12-venv`).

Run the app:

```bash
streamlit run app.py
```

It opens in a browser at the local Streamlit URL. Demo Mode works without an
ESP32, INA219, servo, camera, or trained model.

## Demo Mode / college demonstration

1. Open **Overview** and observe voltage, current, power, energy, source labels,
   charts, and recent activity.
2. Use **Demo Scenario Panel** to switch among Normal, Dust, Crack, Partial
   Shading, Low Solar Generation, Cleaning Successful, Sensor Disconnected, and
   ESP32 Disconnected demonstrations.
3. Open **AI Inspection**, upload an image (or capture from an available webcam),
   choose a demo condition, then run Demo AI Inspection. Results and records
   explicitly say Demo AI / simulated.
4. Open **Automatic Cleaning**. Dust can be cleaned; Crack, Partial Shading,
   Normal, and unknown states are blocked by backend safety checks.
5. Browse **Live Monitoring**, **Historical Data**, and **Alerts & Logs**.
   Export CSV from the historical page and acknowledge/archive alerts.
6. Use **Settings → Reset Demo State** to restore the initial state without
   deleting history. Deleting demo records requires explicit confirmation.

## Hardware Mode preparation

In **Settings**, select Hardware Mode, enter a serial port (`/dev/ttyUSB0`,
`/dev/ttyACM0`, or `COM3` as appropriate), choose baud rate (default 115200),
and save. Use **Check ESP32 connection** to check the port. The app remains
usable while disconnected and displays no simulated measurements in Hardware
Mode. Webcam capture remains a laptop-side OpenCV operation. A trained AI model
is not included; configure your own inference integration before claiming real
AI predictions.

Electrical address and limits, sample interval, confidence threshold, cleaning
timeout, camera index, servo movement range, database, and image folder are
configurable in Settings. Validate physical travel and mechanical safety before
connecting a brush to any motor.

## Serial protocol

Messages are UTF-8 JSON, one object per line. Sensor currents are in amperes.
See [protocol_documentation.md](./protocol_documentation.md) and
[esp32_firmware/protocol_example/protocol_example.ino](./esp32_firmware/protocol_example/protocol_example.ino).
The firmware sketch is a protocol example, not ready-to-flash production motor
control: implement and bench-test INA219 reads, servo drivers, travel limits,
timeouts, and physical emergency stop before use.

## Tests

```bash
python -m pytest
```

Tests cover calculations, time-aware energy integration, sensor simulation,
AI interface, cleaning safety, serial parsing, SQLite persistence, and CSV
export. Streamlit can be started without hardware; Hardware Mode disconnect
handling is part of the application shell.

## Troubleshooting

- **ESP32 disconnected:** verify port permissions, cable, selected port, baud
  rate, and that no other application owns the serial port.
- **Camera unavailable:** select the correct device index or upload an image.
- **No inspection result:** choose a sample condition and click Run Demo AI
  Inspection; the result is simulated unless a real model is integrated.
- **No Hardware readings:** firmware must send valid sensor JSON lines with
  voltage in V, current in A, power in W, and a consistent power calculation.
- **Resetting data:** reset demo state preserves history; Clear Demo Data in
  Settings removes only records whose source is marked Demo.
