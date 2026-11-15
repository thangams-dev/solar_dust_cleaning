#include <ESP32Servo.h>

#define SERVO_PIN 18

Servo s;
int pos = 90;

void moveTo(int target)
{
    while (pos != target)
    {
        pos += (target > pos) ? 1 : -1;
        s.write(pos);
        delay(20);
    }
    Serial.print("MOVE_DONE,");
    Serial.println(pos);
}

void setup()
{
    Serial.begin(115200);
    ESP32PWM::allocateTimer(0);
    s.setPeriodHertz(50);
    s.attach(SERVO_PIN, 500, 2500);
    s.write(pos);
    Serial.println("READY: type an angle 0-180, then Enter");
}

void loop()
{
    if (!Serial.available())
    {
        return;
    }

    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.length() == 0 || !isDigit(line.charAt(0)))
    {
        Serial.println("BAD_INPUT");
        return;
    }

    moveTo(constrain(line.toInt(), 0, 180));
}
