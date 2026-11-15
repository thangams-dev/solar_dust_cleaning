/*
  Solar Panel Cleaner - ESP32 firmware v2
  Hardware: ESP32, INA219 (I2C), MG996R servo (wiper-style brush arm)

  Role: sensor + cleaner only. The laptop makes every decision.

  Laptop -> ESP32 commands (end each with newline):
    CLEAN      run a full cleaning (checks cooldown and hourly limit)
    TEST       one park->far->park sweep, for calibration (no limits)
    PARK       move slowly to the park angle
    ANGLE n    move slowly to angle n (0-180), for calibration
    ABORT      stop the sweep and return to park
    STATUS     print one status line

  ESP32 -> laptop lines:
    READY,v2
    DATA,volt_V,current_mA,power_mW,avg10s_power_mW,STATE
    EVT,CLEAN_START,passes | EVT,CLEAN_DONE | EVT,TEST_DONE
    EVT,CLEAN_ABORT,reason | EVT,MOVE_DONE,angle
    REJECT,BUSY | REJECT,COOLDOWN | REJECT,LIMIT
    WARN,NO_LOAD | ERR,INA219 | ERR,BAD_ARG | UNKNOWN

  Hardware reminders:
    - Power the servo from its own 5-6 V, 2 A supply (not the ESP32 pin).
    - Join the grounds of ESP32, servo supply and INA219.
    - Add a 470-1000 uF capacitor across the servo supply.
    - The solar panel must drive a load through the INA219 (load resistor),
      or current reads zero.
*/

#include <Wire.h>
#include <Adafruit_INA219.h>
#include <ESP32Servo.h>

// ---------- Pins ----------
#define SERVO_PIN 18
#define I2C_SDA   21
#define I2C_SCL   22

// ---------- Servo calibration: CHECK ON YOUR MECHANISM ----------
// Use "ANGLE n" to find angles where the brush rests OFF the glass at both
// ends. MG996R often moves only ~120 deg on 1000-2000 us, ~180 deg on 500-2500 us.
#define SERVO_MIN_US    500
#define SERVO_MAX_US    2500
#define PARK_ANGLE      5      // brush resting beside the panel, off the glass
#define FAR_ANGLE       175    // opposite end of the sweep, off the glass
#define SERVO_STEP_DEG  2      // degrees per step (gentle motion)
#define SERVO_STEP_MS   20     // time per step

// ---------- Cleaning safety limits ----------
#define CLEAN_PASSES        3             // one pass = park -> far -> park
#define CLEAN_TIMEOUT_MS    40000UL       // abort if a sweep takes longer
#define MIN_CLEAN_GAP_MS    120000UL      // cooldown between cleanings
#define MAX_CLEANS_PER_HOUR 3
#define HOUR_MS             3600000UL

// ---------- Sensor ----------
#define SAMPLE_INTERVAL_MS 1000
#define AVG_SAMPLES        10             // moving average = last 10 s
#define NO_LOAD_MIN_V      1.0
#define NO_LOAD_MAX_MA     0.5
#define WARN_REPEAT_MS     30000UL
#define ERR_REPEAT_MS      5000UL

Adafruit_INA219 ina219;
Servo cleaningServo;

enum SystemState { IDLE, CLEANING };
SystemState systemState = IDLE;

// Sensor data
bool sensorReady = false;
float voltage = 0, current = 0, power = 0, avgPower = 0;
float powerBuf[AVG_SAMPLES];
int powerIdx = 0, powerCount = 0;
unsigned long lastSampleTime = 0, lastWarnTime = 0, lastErrTime = 0;

// Servo motion
int currentAngle = PARK_ANGLE;
int targetAngle = PARK_ANGLE;
unsigned long lastStepTime = 0;
bool manualMove = false;

// Sweep state
int passesWanted = 0, passesDone = 0;
bool goingFar = true;
bool sweepCounted = false;
unsigned long sweepStart = 0;

// Safety limits
bool hasCleaned = false;
unsigned long lastCleanEnd = 0;
unsigned long cleanTimes[MAX_CLEANS_PER_HOUR];
int cleanCount = 0;

String inputCommand = "";

// Prototypes
void readSerial();
void processCommand(String cmd);
void updateSensor();
void updateServo();
bool stepServo();
void startSweep(int passes, bool counted);
void finishSweep();
void abortSweep(const char* reason);
const char* cleanBlockReason();
void purgeCleanTimes();
void printStatus();

void setup()
{
    Serial.begin(115200);
    Wire.begin(I2C_SDA, I2C_SCL);

    ESP32PWM::allocateTimer(0);
    cleaningServo.setPeriodHertz(50);
    cleaningServo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
    // If power was lost mid-sweep the brush may be on the glass; the servo
    // cannot know its position, so this move to park is not gentle.
    cleaningServo.write(PARK_ANGLE);

    if (ina219.begin())
    {
        ina219.setCalibration_32V_2A();
        sensorReady = true;
    }

    Serial.println("READY,v2");
    if (!sensorReady)
    {
        Serial.println("ERR,INA219");
    }
}

void loop()
{
    readSerial();
    updateSensor();
    updateServo();
}

// ---------------- Serial input ----------------
void readSerial()
{
    while (Serial.available())
    {
        char received = Serial.read();

        if (received == '\n')
        {
            inputCommand.trim();
            inputCommand.toUpperCase();
            processCommand(inputCommand);
            inputCommand = "";
        }
        else if (received != '\r')
        {
            inputCommand += received;
            if (inputCommand.length() > 40)
            {
                inputCommand = "";
            }
        }
    }
}

void processCommand(String cmd)
{
    if (cmd == "CLEAN")
    {
        const char* reason = cleanBlockReason();
        if (reason != nullptr)
        {
            Serial.print("REJECT,");
            Serial.println(reason);
            return;
        }
        startSweep(CLEAN_PASSES, true);
    }
    else if (cmd == "TEST")
    {
        if (systemState != IDLE)
        {
            Serial.println("REJECT,BUSY");
            return;
        }
        startSweep(1, false);
    }
    else if (cmd == "PARK")
    {
        if (systemState != IDLE)
        {
            Serial.println("REJECT,BUSY");
            return;
        }
        targetAngle = PARK_ANGLE;
        manualMove = true;
    }
    else if (cmd.startsWith("ANGLE "))
    {
        if (systemState != IDLE)
        {
            Serial.println("REJECT,BUSY");
            return;
        }
        String arg = cmd.substring(6);
        arg.trim();
        if (arg.length() == 0 || !isDigit(arg.charAt(0)))
        {
            Serial.println("ERR,BAD_ARG");
            return;
        }
        targetAngle = constrain(arg.toInt(), 0, 180);
        manualMove = true;
    }
    else if (cmd == "ABORT")
    {
        if (systemState == CLEANING)
        {
            abortSweep("USER");
        }
    }
    else if (cmd == "STATUS")
    {
        printStatus();
    }
    else if (cmd.length() > 0)
    {
        Serial.println("UNKNOWN");
    }
}

// ---------------- Sensor ----------------
void updateSensor()
{
    unsigned long now = millis();

    if (now - lastSampleTime < SAMPLE_INTERVAL_MS)
    {
        return;
    }
    lastSampleTime = now;

    if (!sensorReady)
    {
        if (now - lastErrTime >= ERR_REPEAT_MS)
        {
            lastErrTime = now;
            Serial.println("ERR,INA219");
        }
        return;
    }

    voltage = ina219.getBusVoltage_V();
    current = ina219.getCurrent_mA();
    power = ina219.getPower_mW();

    // Moving average of the last AVG_SAMPLES readings.
    // Wait at least 10 s after cleaning before reading "after" power.
    powerBuf[powerIdx] = power;
    powerIdx = (powerIdx + 1) % AVG_SAMPLES;
    if (powerCount < AVG_SAMPLES)
    {
        powerCount++;
    }
    float sum = 0;
    for (int i = 0; i < powerCount; i++)
    {
        sum += powerBuf[i];
    }
    avgPower = sum / powerCount;

    Serial.print("DATA,");
    Serial.print(voltage, 3);
    Serial.print(",");
    Serial.print(current, 2);
    Serial.print(",");
    Serial.print(power, 2);
    Serial.print(",");
    Serial.print(avgPower, 2);
    Serial.print(",");
    Serial.println(systemState == CLEANING ? "CLEANING" : "IDLE");

    // Heuristic: voltage present but no current -> panel not driving a load
    if (voltage > NO_LOAD_MIN_V && current < NO_LOAD_MAX_MA &&
        now - lastWarnTime >= WARN_REPEAT_MS)
    {
        lastWarnTime = now;
        Serial.println("WARN,NO_LOAD");
    }
}

// ---------------- Servo and sweep ----------------
bool stepServo()
{
    if (currentAngle == targetAngle)
    {
        return true;
    }

    unsigned long now = millis();
    if (now - lastStepTime < SERVO_STEP_MS)
    {
        return false;
    }
    lastStepTime = now;

    int diff = targetAngle - currentAngle;
    if (abs(diff) <= SERVO_STEP_DEG)
    {
        currentAngle = targetAngle;
    }
    else
    {
        currentAngle += (diff > 0) ? SERVO_STEP_DEG : -SERVO_STEP_DEG;
    }
    cleaningServo.write(currentAngle);

    return currentAngle == targetAngle;
}

void updateServo()
{
    if (systemState == IDLE)
    {
        // Manual moves (PARK, ANGLE) and the return after an abort
        if (stepServo() && manualMove)
        {
            manualMove = false;
            Serial.print("EVT,MOVE_DONE,");
            Serial.println(currentAngle);
        }
        return;
    }

    // CLEANING
    if (millis() - sweepStart > CLEAN_TIMEOUT_MS)
    {
        abortSweep("TIMEOUT");
        return;
    }

    if (!stepServo())
    {
        return;
    }

    // Reached the current target
    if (goingFar)
    {
        goingFar = false;
        targetAngle = PARK_ANGLE;
    }
    else
    {
        passesDone++;
        if (passesDone >= passesWanted)
        {
            finishSweep();
        }
        else
        {
            goingFar = true;
            targetAngle = FAR_ANGLE;
        }
    }
}

void startSweep(int passes, bool counted)
{
    systemState = CLEANING;
    passesWanted = passes;
    passesDone = 0;
    goingFar = true;
    targetAngle = FAR_ANGLE;
    sweepCounted = counted;
    sweepStart = millis();

    if (counted)
    {
        purgeCleanTimes();
        if (cleanCount < MAX_CLEANS_PER_HOUR)
        {
            cleanTimes[cleanCount++] = sweepStart;
        }
    }

    Serial.print("EVT,CLEAN_START,");
    Serial.println(passes);
}

void finishSweep()
{
    systemState = IDLE;
    if (sweepCounted)
    {
        hasCleaned = true;
        lastCleanEnd = millis();
        Serial.println("EVT,CLEAN_DONE");
    }
    else
    {
        Serial.println("EVT,TEST_DONE");
    }
}

void abortSweep(const char* reason)
{
    systemState = IDLE;
    targetAngle = PARK_ANGLE;   // returns slowly; EVT,MOVE_DONE when parked
    manualMove = true;
    if (sweepCounted)
    {
        hasCleaned = true;
        lastCleanEnd = millis();
    }
    Serial.print("EVT,CLEAN_ABORT,");
    Serial.println(reason);
}

// ---------------- Safety limits ----------------
void purgeCleanTimes()
{
    while (cleanCount > 0 && millis() - cleanTimes[0] >= HOUR_MS)
    {
        for (int i = 1; i < cleanCount; i++)
        {
            cleanTimes[i - 1] = cleanTimes[i];
        }
        cleanCount--;
    }
}

const char* cleanBlockReason()
{
    if (systemState != IDLE)
    {
        return "BUSY";
    }
    if (hasCleaned && millis() - lastCleanEnd < MIN_CLEAN_GAP_MS)
    {
        return "COOLDOWN";
    }
    purgeCleanTimes();
    if (cleanCount >= MAX_CLEANS_PER_HOUR)
    {
        return "LIMIT";
    }
    return nullptr;
}

void printStatus()
{
    purgeCleanTimes();

    unsigned long cooldownLeft = 0;
    if (hasCleaned && millis() - lastCleanEnd < MIN_CLEAN_GAP_MS)
    {
        cooldownLeft = MIN_CLEAN_GAP_MS - (millis() - lastCleanEnd);
    }

    Serial.print("STATUS,");
    Serial.print(systemState == CLEANING ? "CLEANING" : "IDLE");
    Serial.print(",angle=");
    Serial.print(currentAngle);
    Serial.print(",V=");
    Serial.print(voltage, 3);
    Serial.print(",mA=");
    Serial.print(current, 2);
    Serial.print(",mW=");
    Serial.print(power, 2);
    Serial.print(",avg_mW=");
    Serial.print(avgPower, 2);
    Serial.print(",cleans_last_hour=");
    Serial.print(cleanCount);
    Serial.print(",cooldown_ms=");
    Serial.println(cooldownLeft);
}
