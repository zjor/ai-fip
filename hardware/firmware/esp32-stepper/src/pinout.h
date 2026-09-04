#include <Arduino.h>

// MPU9250 wiring
#define PIN_IMU_SCL         GPIO_NUM_22
#define PIN_IMU_SDA         GPIO_NUM_21
#define PIN_IMU_INT         GPIO_NUM_19

// Stepper 1 wiring
#define PIN_STEPPER_EN      GPIO_NUM_15
#define PIN_STEPPER_DIR     GPIO_NUM_14
#define PIN_STEPPER_STEP    GPIO_NUM_12