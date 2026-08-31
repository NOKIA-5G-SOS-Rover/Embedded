#include "motordriver.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include "Arduino_RouterBridge.h"

const int LEFT_DIR_PIN = 4;
const int LEFT_PWM_PIN = 5;
const int RIGHT_DIR_PIN = 7;
const int RIGHT_PWM_PIN = 6;

// ---------- Optional ultrasonic (HC-SR04) safety sensor ----------
// Set to 1 once you've wired an HC-SR04: VCC->5V, GND->GND, TRIG->D2, ECHO->D3
// (through a voltage divider on ECHO if your board's logic is 3.3V - the
// HC-SR04's echo output is 5V and can damage a 3.3V input pin otherwise).
#define HAS_ULTRASONIC 0
const int ULTRASONIC_TRIG_PIN = 2;
const int ULTRASONIC_ECHO_PIN = 3;

Rover myRover(LEFT_DIR_PIN, LEFT_PWM_PIN, RIGHT_DIR_PIN, RIGHT_PWM_PIN);
Adafruit_MPU6050 mpu;

float gyroXOffset = 0.0;
bool gyroAvailable = false;  // motors/Bridge must work even if the gyro doesn't

// ---------- Soft-start ramp state machine ----------
// PROBLEM THIS SOLVES: jumping straight to a high PWM value from a dead
// stop forces peak stall current through the motors, driver, and battery
// all at once - this is almost certainly what's browning out the board at
// high speed. Ramping the PWM up gradually instead spreads that current
// draw out over a couple hundred ms instead of a single instant spike.
// It also makes Bridge.provide functions return immediately (they just set
// a target, they don't drive the motors directly), which matters for
// command-flooding robustness - see main.py's rewritten command handling.

enum DriveCommand { CMD_STOP, CMD_FORWARD, CMD_BACKWARD, CMD_TURN_LEFT, CMD_TURN_RIGHT, CMD_ARC_LEFT, CMD_ARC_RIGHT, CMD_REVERSE_ARC_LEFT, CMD_REVERSE_ARC_RIGHT };

DriveCommand targetCommand = CMD_STOP;
int targetSpeed = 0;

DriveCommand appliedCommand = CMD_STOP;
int rampedSpeed = 0;

const int RAMP_STEP = 12;           // PWM units per tick - tune down for a gentler ramp
const int RAMP_INTERVAL_MS = 15;    // ~255/12*15ms =~ 320ms for a full 0->255 ramp
unsigned long lastRampTick = 0;

void applyMotorCommand(DriveCommand cmd, int speed) {
  switch (cmd) {
    case CMD_FORWARD:    myRover.forward(speed); break;
    case CMD_BACKWARD:   myRover.backward(speed); break;
    case CMD_TURN_LEFT:  myRover.turnLeft(speed); break;
    case CMD_TURN_RIGHT: myRover.turnRight(speed); break;
    case CMD_ARC_LEFT:          myRover.driveLeft(speed); break;
    case CMD_ARC_RIGHT:         myRover.driveRight(speed); break;
    case CMD_REVERSE_ARC_LEFT:  myRover.reverseArcLeft(speed); break;
    case CMD_REVERSE_ARC_RIGHT: myRover.reverseArcRight(speed); break;
    case CMD_STOP:              myRover.stop(); break;
  }
}

void rampTick() {
  if (millis() - lastRampTick < RAMP_INTERVAL_MS) return;
  lastRampTick = millis();

  // STOP always cuts immediately - safety stops should never be delayed by
  // a smooth deceleration, that defeats the point of a stop command.
  if (targetCommand == CMD_STOP) {
    if (appliedCommand != CMD_STOP || rampedSpeed != 0) {
      appliedCommand = CMD_STOP;
      rampedSpeed = 0;
      applyMotorCommand(CMD_STOP, 0);
    }
    return;
  }

  // Changing direction/mode: ramp DOWN to 0 first before switching, so we
  // never instantly reverse a spinning motor (also a big current spike).
  if (appliedCommand != targetCommand) {
    if (rampedSpeed > 0) {
      rampedSpeed = max(0, rampedSpeed - RAMP_STEP);
      applyMotorCommand(appliedCommand, rampedSpeed);
      if (rampedSpeed == 0) applyMotorCommand(CMD_STOP, 0); // fully release the old direction
    } else {
      appliedCommand = targetCommand; // now safe to switch, ramp-up starts next tick
    }
    return;
  }

  // Same direction, ramp toward target speed (up or down)
  if (rampedSpeed < targetSpeed) {
    rampedSpeed = min(targetSpeed, rampedSpeed + RAMP_STEP);
    applyMotorCommand(appliedCommand, rampedSpeed);
  } else if (rampedSpeed > targetSpeed) {
    rampedSpeed = max(targetSpeed, rampedSpeed - RAMP_STEP);
    applyMotorCommand(appliedCommand, rampedSpeed);
  }
}

// ---------- Bridge-exposed drive commands ----------
// Each of these just sets a target and returns immediately - actual motor
// control happens in loop()'s rampTick(). This keeps every Bridge call fast
// even under a flood of rapid commands, and gives every motion a soft start.

// Temporary safety ceiling while the >95% brownout gets root-caused (likely
// wiring gauge/connector/battery C-rating, not something software can fully
// fix) - clamps the highest allowed PWM regardless of what's requested.
// Raise this back toward 255 once the hardware side is confirmed solid.
const int MAX_SAFE_PWM = 220; // ~86% of 255

int clampToSafeSpeed(int requested) {
  return constrain(requested, 0, MAX_SAFE_PWM);
}

bool cmd_forward(int speed)    { targetCommand = CMD_FORWARD;    targetSpeed = clampToSafeSpeed(speed); return true; }
bool cmd_backward(int speed)   { targetCommand = CMD_BACKWARD;   targetSpeed = clampToSafeSpeed(speed); return true; }
bool cmd_turn_left(int speed)  { targetCommand = CMD_TURN_LEFT;  targetSpeed = clampToSafeSpeed(speed); return true; }
bool cmd_turn_right(int speed) { targetCommand = CMD_TURN_RIGHT; targetSpeed = clampToSafeSpeed(speed); return true; }
bool cmd_arc_left(int speed)          { targetCommand = CMD_ARC_LEFT;          targetSpeed = clampToSafeSpeed(speed); return true; }
bool cmd_arc_right(int speed)         { targetCommand = CMD_ARC_RIGHT;         targetSpeed = clampToSafeSpeed(speed); return true; }
bool cmd_reverse_arc_left(int speed)  { targetCommand = CMD_REVERSE_ARC_LEFT;  targetSpeed = clampToSafeSpeed(speed); return true; }
bool cmd_reverse_arc_right(int speed) { targetCommand = CMD_REVERSE_ARC_RIGHT; targetSpeed = clampToSafeSpeed(speed); return true; }

bool cmd_stop() {
  targetCommand = CMD_STOP;
  targetSpeed = 0;
  return true;
}

bool cmd_turn_degrees(float degrees, int speed) {
  if (!gyroAvailable) {
    Serial.println("cmd_turn_degrees called but gyro is unavailable - ignoring");
    return false;
  }
  myRover.turnByDegrees(degrees, speed);
  return true;
}

// ---------- Battery voltage divider ----------
// Wire: Battery+ -> R1 -> (ADC pin) -> R2 -> GND. R1/R2 values below MUST
// match your actual resistors, and must be sized so max battery voltage
// never exceeds this board's ADC reference (3.3V) at the ADC pin itself -
// check that before wiring anything to the battery.
const int BATTERY_ADC_PIN = A0;

float get_battery_raw() {
  int reading = analogRead(BATTERY_ADC_PIN);         // native ADC resolution, board-dependent
  return map(reading, 0, 1023, 0, 255);               // TODO: confirm this board's native ADC max - 1023 assumes 10-bit; adjust if it's actually 12-bit (4095)
}

// Returns distance in cm, or -1 if no ultrasonic sensor is wired/enabled.
// The Linux-side autonomy loop should treat -1 as "no safety sensor available"
// and fall back to being extra conservative (or refuse to run autonomously).
float get_distance_cm() {
#if HAS_ULTRASONIC
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG_PIN, LOW);

  long duration = pulseIn(ULTRASONIC_ECHO_PIN, HIGH, 30000); // 30ms timeout (~5m range)
  if (duration == 0) return -1; // timed out, no echo
  return duration * 0.0343 / 2.0; // speed of sound conversion
#else
  return -1;
#endif
}

void setup() {
  Serial.begin(115200);
  myRover.begin();

  Serial.println("Initializing MPU6050...");
  if (!mpu.begin()) {
    // Previously halted here forever with while(1){delay(10);} - that meant
    // a loose gyro wire bricked ALL connectivity, not just turn-by-degrees.
    // Motors and the Bridge (and therefore network control) must still come
    // up regardless of gyro status.
    Serial.println("Failed to find MPU6050 chip! Continuing WITHOUT gyro - turn-by-degrees will be disabled.");
    gyroAvailable = false;
  } else {
    gyroAvailable = true;
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    Serial.println("\nCalibrating Gyroscope... DO NOT MOVE ROVER!");

    float totalDrift = 0.0;
    int numSamples = 200;

    for (int i = 0; i < numSamples; i++) {
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);
      totalDrift += (g.gyro.x * 57.2958);
      delay(10);
    }

    gyroXOffset = totalDrift / numSamples;

    Serial.print("Calibration complete! X-Axis Offset: ");
    Serial.print(gyroXOffset);
    Serial.println(" deg/s\n");
  }

  // Register every drive command so the Linux side can call it by name
  Bridge.begin();
  Bridge.provide("cmd_forward", cmd_forward);
  Bridge.provide("cmd_backward", cmd_backward);
  Bridge.provide("cmd_stop", cmd_stop);
  Bridge.provide("cmd_turn_left", cmd_turn_left);
  Bridge.provide("cmd_turn_right", cmd_turn_right);
  Bridge.provide("cmd_arc_left", cmd_arc_left);
  Bridge.provide("cmd_arc_right", cmd_arc_right);
  Bridge.provide("cmd_reverse_arc_left", cmd_reverse_arc_left);
  Bridge.provide("cmd_reverse_arc_right", cmd_reverse_arc_right);
  Bridge.provide("cmd_turn_degrees", cmd_turn_degrees);
  Bridge.provide("get_distance_cm", get_distance_cm);
  Bridge.provide("get_battery_raw", get_battery_raw);

#if HAS_ULTRASONIC
  pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
  pinMode(ULTRASONIC_ECHO_PIN, INPUT);
#endif

  Serial.println("Ready. Listening for Bridge commands from Linux side.");
}

void loop() {
  rampTick();
}
