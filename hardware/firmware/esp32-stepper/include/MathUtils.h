#include <math.h>
#include <Arduino.h>

float clamp(float value, float maxValue) {
  if (value > maxValue) {
    return maxValue;
  } else if (value < -maxValue) {
    return -maxValue;
  } else {
    return value;
  }
}