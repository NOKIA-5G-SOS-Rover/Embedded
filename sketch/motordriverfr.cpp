#include "motordriverfr.h"

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Motor::Motor(int dir, int pwm) {
    dirPin = dir;
    pwmPin = pwm;
}

void Motor::begin() {
    pinMode(dirPin, OUTPUT);
    pinMode(pwmPin, OUTPUT);
    stop();
}

void Motor::drive(int speed) {
    speed = constrain(speed, -255, 255);
    if (speed >= 0) {
        digitalWrite(dirPin, HIGH);
        analogWrite(pwmPin, speed);
    } else {
        digitalWrite(dirPin, LOW);
        analogWrite(pwmPin, -speed);
    }
}

void Motor::stop() {
    analogWrite(pwmPin, 0);
}

Rover::Rover(int leftDir, int leftPwm, int rightDir, int rightPwm) 
    : leftMotor(leftDir, leftPwm), rightMotor(rightDir, rightPwm) {}

void Rover::begin() {
    leftMotor.begin();
    rightMotor.begin();
}

void Rover::turnRight(int speed) {
    leftMotor.drive(speed);
    rightMotor.drive(speed); 
}

void Rover::turnLeft(int speed) {
    leftMotor.drive(-speed);
    rightMotor.drive(-speed);
}

void Rover::driveLeft(int speed) {
    leftMotor.drive(-speed);
    rightMotor.stop();
}

void Rover::driveRight(int speed) {
    leftMotor.stop();
    rightMotor.drive(speed); 
}

void Rover::forward(int speed) {
    leftMotor.drive(-speed); 
    rightMotor.drive(speed);  
}

void Rover::backward(int speed) {
    leftMotor.drive(speed);   
    rightMotor.drive(-speed); 
}

void Rover::stop() {
    leftMotor.stop();
    rightMotor.stop();
}

void Rover::turnByDegrees(float targetAngle, int speed) {
    extern Adafruit_MPU6050 mpu;
    extern float gyroXOffset; 
    
    float currentAngle = 0.0;
    unsigned long lastTime = millis();

    // Because we are actively braking, the rover will coast much less.
    // You can likely lower this offset compared to the coasting method.
    float stopOffset = 15.0; 

    Serial.print("\n--- STARTING TURN: ");
    Serial.print(targetAngle);
    Serial.println(" DEGREES ---");

    if (targetAngle == 0) return;
    bool turningRight = (targetAngle > 0);

    if (turningRight) {
        turnRight(speed);
    } else {
        turnLeft(speed); 
    }

    while (currentAngle < (abs(targetAngle) - stopOffset)) {
        
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);

        unsigned long currentTime = millis();
        float dt = (currentTime - lastTime) / 1000.0;
        lastTime = currentTime;

        float rawXRotation = g.gyro.x * 57.2958;
        float trueXRotation = rawXRotation - gyroXOffset;

        currentAngle += abs(trueXRotation * dt);

        Serial.print("Target: ");
        Serial.print(abs(targetAngle));
        Serial.print(" | Current Angle: ");
        Serial.print(currentAngle);
        Serial.print(" | True Speed: ");
        Serial.println(abs(trueXRotation));

        delay(10);
    }
    
    // --- NEW: ACTIVE BRAKING SEQUENCE ---
    // Instantly throw the motors in reverse to kill all forward momentum
    int brakeTime = 40; // Milliseconds to hold the brake. 
    
    if (turningRight) {
        turnLeft(speed); 
    } else {
        turnRight(speed); 
    }
    
    delay(brakeTime); // Hold the brake for a fraction of a second
    stop();           // Cut power completely
    // ------------------------------------

    Serial.println("--- TURN COMPLETE ---\n");
}