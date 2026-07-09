#include "motordriverchiardriver.h"

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

// void Rover::turnByDegrees(int angle, int speed) {
//     unsigned long turnTime = abs(angle) * msPerDegree;
    
//     if (angle > 0) {
//         turnRight(speed);
//     } else if (angle < 0) {
//         turnLeft(speed); 
//     }
    
//     delay(turnTime);
//     stop();
// }