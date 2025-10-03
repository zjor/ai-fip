#include <math.h>
#include <Arduino.h>

inline float normalizeAngle(float value) {
  return ((value < 180) ? value : value - 360.0f) * DEG_TO_RAD;
}

float clamp(float value, float maxValue) {
  if (value > maxValue) {
    return maxValue;
  } else if (value < -maxValue) {
    return -maxValue;
  } else {
    return value;
  }
}