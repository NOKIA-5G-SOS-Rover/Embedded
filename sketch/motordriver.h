#pragma once 
#include <Arduino.h> 

class Motor {
  private:
    int dirPin;
    int pwmPin;

  public:
    Motor(int dir, int pwm);
    void begin();
    void drive(int speed);
    void stop();
};

class Rover {
  private:
    Motor leftMotor;
    Motor rightMotor;
    float msPerDegree = 8.5; 

  public:
    Rover(int leftDir, int leftPwm, int rightDir, int rightPwm);
    void begin();
    void turnRight(int speed);
    void turnLeft(int speed);
    void driveLeft(int speed);
    void driveRight(int speed);
    void reverseArcLeft(int speed);
    void reverseArcRight(int speed);
    void forward(int speed);
    void backward(int speed);
    void stop();
    void turnByDegrees(float angle, int speed);

};