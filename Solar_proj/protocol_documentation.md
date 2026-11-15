# ESP32 newline-delimited JSON protocol

USB serial is UTF-8, one JSON object per line, terminated by `\n`. Default baud
rate is 115200 (configurable in Settings). The ESP32 owns all sensor and servo
measurements; the dashboard never substitutes simulated measurements in
Hardware Mode.

## ESP32 to laptop

Sensor message (current is amperes; power is watts):

```json
{"type":"sensor","voltage":5.8,"current":0.145,"power":0.841,"timestamp":"2026-01-01T10:00:00+00:00"}
```

Cleaning progress:

```json
{"type":"cleaning_status","status":"RUNNING"}
{"type":"cleaning_status","status":"COMPLETED"}
```

Accepted cleaning statuses are `IDLE`, `RUNNING`, `COMPLETED`, `FAILED`, and
`STOPPED`. Errors use `{"type":"error","message":"Servo movement failed"}`.
Sensor messages require finite, non-negative voltage/current/power; reported
power must agree with voltage × current within tolerance.

## Laptop to ESP32

```json
{"command":"START_CLEANING"}
{"command":"STOP_CLEANING"}
```

The laptop sends commands only after the backend safety check permits cleaning
(dust condition and sufficient confidence). Crack, partial shading, normal, or
unknown conditions are rejected. Hardware cleaning is only recorded as
completed after the ESP32 sends `COMPLETED`.

## Hardware integration notes

Connect INA219 to ESP32 over I2C and report current in amperes. Connect the
servo signal to a suitable ESP32 PWM-capable pin. Power the servo from a
regulated supply appropriate to that servo, not the ESP32 3.3 V pin, and share
ground between the supply and ESP32. Define and validate mechanical movement
limits before enabling servo movement. Keep the laptop USB connection and
webcam independent from the sensor/servo power supply.
