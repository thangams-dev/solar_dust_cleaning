#include <ESP32Servo.h>

#define SERVO_PIN 18

Servo cleaning_servo;

void setup()
{
    Serial.begin(115200);

    cleaning_servo.setPeriodHertz(50);
    cleaning_servo.attach(SERVO_PIN, 1000, 2000);

    Serial.println("Servo test started");
}

void loop()
{
    cleaning_servo.write(30);
    Serial.println("Position: 30 degrees");
    delay(2000);

    cleaning_servo.write(90);
    Serial.println("Position: 90 degrees");
    delay(2000);

    cleaning_servo.write(150);
    Serial.println("Position: 150 degrees");
    delay(2000);
}