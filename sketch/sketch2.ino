#include "motordriverfr.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

const int LEFT_DIR_PIN = 4;
const int LEFT_PWM_PIN = 5;
const int RIGHT_DIR_PIN = 7;
const int RIGHT_PWM_PIN = 6;

Rover myRover(LEFT_DIR_PIN, LEFT_PWM_PIN, RIGHT_DIR_PIN, RIGHT_PWM_PIN);
Adafruit_MPU6050 mpu; 

// NEW: Global variable to hold our calculated drift error
float gyroXOffset = 0.0;

const int TEST_SPEED = 130; 

void setup() {
  Serial.begin(115200);
  myRover.begin();

  Serial.println("Initializing MPU6050...");
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip! Check wiring.");
    while (1) { delay(10); } 
  }
  
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  // --- AUTOMATIC CALIBRATION ROUTINE ---
  Serial.println("\nCalibrating Gyroscope... DO NOT MOVE ROVER!");
  
  float totalDrift = 0.0;
  int numSamples = 200; // Take 200 readings
  
  for (int i = 0; i < numSamples; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    // Add the current reading to our running total
    totalDrift += (g.gyro.x * 57.2958);
    delay(10); // Wait 10ms between readings
  }
  
  // Divide the total by the number of samples to find the average error
  gyroXOffset = totalDrift / numSamples;
  
  Serial.print("Calibration complete! X-Axis Offset: ");
  Serial.print(gyroXOffset);
  Serial.println(" deg/s\n");
  // -------------------------------------

  Serial.println("Starting turns in 3 seconds...");
  delay(3000);
}

void loop() {
  // Turn 90 degrees Right
  myRover.turnByDegrees(90.0, TEST_SPEED);
  delay(3000); 

  // Turn 90 degrees Left
  myRover.turnByDegrees(-90.0, TEST_SPEED);
  
  Serial.println("Waiting 5 seconds before repeating...");
  delay(5000);
}