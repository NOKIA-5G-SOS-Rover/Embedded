#include "motordriverchiardriver.h" 

Rover myRover(4, 5, 7, 6); 

void setup() {
  myRover.begin();
  delay(2000); 
}

void loop() {
  int testSpeed = 150; 

  myRover.forward(testSpeed);
  delay(2000);
  myRover.stop();
  delay(1000);

  myRover.backward(testSpeed);
  delay(2000);
  myRover.stop();
  delay(1000);

  myRover.turnLeft(testSpeed);
  delay(1000);
  myRover.stop();
  delay(1000);

  myRover.turnRight(testSpeed);
  delay(1000);
  myRover.stop();
  delay(1000);

  myRover.driveLeft(testSpeed);
  delay(2000);
  myRover.stop();
  delay(1000);

  myRover.driveRight(testSpeed);
  delay(2000);
  myRover.stop();

  //myRover.turnByDegrees(90, testSpeed); 
  //delay(1000);

  //myRover.turnByDegrees(-180, testSpeed);
  
  while(true) {
    myRover.stop();
  }
}