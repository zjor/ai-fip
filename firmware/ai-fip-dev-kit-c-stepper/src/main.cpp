#include <Arduino.h>

int state = LOW;

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  digitalWrite(LED_BUILTIN, state);
  state = !state;
  delay(500);

  Serial.println("Hello world");

}


