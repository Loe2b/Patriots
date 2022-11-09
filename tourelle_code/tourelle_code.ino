#define joystick1    A0
#define joystick2    A1
#define servo1       8
#define servo2       9
#define servo3       2
#define boutonfire   0
#define boutonreset  4

int posx = 90;
int posy = 90;
int posfire = 180;

#include <Servo.h>
Servo PAN;
Servo TILT;
Servo FIRE;

void setup() {
  Serial.begin(9600);
  PAN.attach(servo1); PAN.write(posx);
  TILT.attach(servo2); TILT.write(posy);
  FIRE.attach(servo3); FIRE.write(posfire);
  delay(10);
  pinMode(boutonfire, INPUT_PULLUP);
  pinMode(boutonreset, INPUT_PULLUP);

}

void loop() {
  int Joystick1 = analogRead(joystick1);
  int Joystick2 = analogRead(joystick2);

  if (Joystick1 > 600 && Joystick1 < 1024 && posx < 110) {
    posx++;
    PAN.write(posx);
    delay(10);
  }
  if (Joystick1 >= 0 && Joystick1 < 200 && posx >70) {
    posx--;
    PAN.write(posx);
    delay(10);
  }
  if (Joystick2 > 600 && Joystick2 < 1024 && posy < 180) {
    posy++;
    TILT.write(posy);
    delay(10);
  }
  if (Joystick2 >= 0 && Joystick2 < 200 && posy > 0) {
    posy--;
    TILT.write(posy);
    delay(10);
  }
  if (digitalRead(boutonfire) == 0) {
    FIRE.write(60);
  }
  else {
    FIRE.write(180);
  }
  if (digitalRead(boutonreset) == 0){
    TILT.write(90);
    PAN.write(90);
    }

}
