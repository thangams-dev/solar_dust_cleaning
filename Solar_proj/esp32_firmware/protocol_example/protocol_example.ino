#include <ArduinoJson.h>

// Protocol skeleton only: install ArduinoJson and adapt the INA219/PWM wiring.
// Never connect a servo's power lead to the ESP32 3.3 V output.
static const int SERVO_PIN = 18;
static bool cleaning = false;
static String inputLine;

void setup() {
  Serial.begin(115200);
  pinMode(SERVO_PIN, OUTPUT);
  // Initialize the INA219 and a servo driver/library for your hardware here.
}

void loop() {
  while (Serial.available()) {
    char ch = static_cast<char>(Serial.read());
    if (ch == '\n') {
      StaticJsonDocument<192> command;
      if (deserializeJson(command, inputLine) == DeserializationError::Ok) {
        const char* value = command["command"] | "";
        if (strcmp(value, "START_CLEANING") == 0) {
          cleaning = true;
          Serial.println("{\"type\":\"cleaning_status\",\"status\":\"RUNNING\"}");
        } else if (strcmp(value, "STOP_CLEANING") == 0) {
          cleaning = false;
          Serial.println("{\"type\":\"cleaning_status\",\"status\":\"STOPPED\"}");
        }
      } else {
        Serial.println("{\"type\":\"error\",\"message\":\"Invalid laptop JSON\"}");
      }
      inputLine = "";
    } else if (inputLine.length() < 256) {
      inputLine += ch;
    } else {
      inputLine = "";
      Serial.println("{\"type\":\"error\",\"message\":\"Command line too long\"}");
    }
  }

  // TODO: Read the INA219 and emit one sensor JSON line at the chosen cadence.
  // TODO: Apply mechanically safe servo motion while cleaning is true.
  // TODO: Emit COMPLETED only after the physical cycle finishes successfully.
  delay(2);
}
