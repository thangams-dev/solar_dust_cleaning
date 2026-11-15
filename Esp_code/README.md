# ESP32 PlatformIO Starter

This project targets the ESP32 Dev Module (`esp32dev`) with the Arduino
framework. The starter firmware blinks the board's `LED_BUILTIN` once per
second.

Build and upload from the project directory with PlatformIO:

```sh
pio run
pio run --target upload
pio device monitor
```

If your board does not define `LED_BUILTIN` or uses a different onboard LED
pin, update the pin in `src/main.cpp` for that board.
