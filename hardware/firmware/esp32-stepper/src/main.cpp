/**
 * AI-controlled flywheel inverted pendulum firmware.
 * 
 * Parts:
 *  - ESP32 DevKit-C
 *  - Nema17 stepper motor
 *  - A4988 stepper driver
 *  - MPU6050
 * 
 * @author Sergey Royz (zjor.se@gmail.com) 
 * @version 0.1
 * @date 2025-05-17 - ...
 */
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Kalman.h>

#include <math.h>

#include "pinout.h"
#include "stepper/stepper.h"
#include "FixedRateExecutor.h"
#include "MPU.h"
#include "MathUtils.h"

// pulses per revolution
#define PPR               1600

#define CPU_FREQ_MHZ      240
#define CPU_FREQ_DIVIDER  120
#define TICKS_PER_SECOND  (200000 * (CPU_FREQ_MHZ / CPU_FREQ_DIVIDER))
#define PULSE_WIDTH       1

/**
 * How to calibrate:
 * - set to zero
 * - get a settled value
 */
#define THETA_BIAS_DEG  -2.00f

// LQR params
#define K_THETA         40000.0
#define K_THETA_DOT     250.0
#define K_PHI           0.0
#define K_PHI_DOT       -110.0

void initMPU();
void initTimerInterrupt();
void updateStepperVelocity(unsigned long);
void updateControl(unsigned long);
void log();

FixedRateExecutor logger(25000, log);

static inline float accelRollDegFrom(const sensors_event_t&);
static inline float gyroRollRadFrom(const sensors_event_t&);

hw_timer_t * timer = NULL;

Stepper stepper(PIN_STEPPER_EN, PIN_STEPPER_DIR, PIN_STEPPER_STEP, TICKS_PER_SECOND, PPR, PULSE_WIDTH);

MPU mpu(THETA_BIAS_DEG);
float last_ms = 0;

// stepper params
float velocity = 0.0;
float accel = 0.0;

void setup() {
  setCpuFrequencyMhz(CPU_FREQ_MHZ);

  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(1000000UL);

  initTimerInterrupt();
  delay(250);
  mpu.init();

  stepper.init();
  stepper.setEnabled(true);
  last_ms = millis();
}

void loop() {
  unsigned long nowMicros = micros();
  updateStepperVelocity(nowMicros);
  updateControl(nowMicros);

  unsigned long now = millis();
  float dt = (now - last_ms) / 1000.0f;
  mpu.updateAngles(dt);
  last_ms = now;

  logger.tick(nowMicros);
}

void updateStepperVelocity(unsigned long nowMicros) {
  static unsigned long timestamp = micros();
  if (nowMicros - timestamp < 50 /* 20 kHz */) {
    return;
  }
  
  float dt = ((float) (nowMicros - timestamp)) * 1e-6;
  velocity += accel * dt;
  stepper.setVelocity(velocity);
  timestamp = nowMicros;
}

void updateControl(unsigned long nowMicros) {
  static unsigned long timestamp = micros();
  if (nowMicros - timestamp < 1000 /* 1kHz*/) {
    return;
  }

  float theta = mpu.getAngleRad();
  theta = (theta > 0) ? theta - M_PI : theta + M_PI;
  if (abs(theta) < M_PI / 16) {
    float theta_dot = mpu.getAngularVelocityRad();
    float phi = 0; //stepper position
    float phi_dot = velocity; //TODO: convert to radians/sec

    accel = K_THETA * theta + 
            K_THETA_DOT * theta_dot + 
            K_PHI * phi + 
            K_PHI_DOT * phi_dot;

    accel = clamp(accel, 150);
  } else {
    accel = 0;
    velocity = 0;
  }
  timestamp = nowMicros;
}

void IRAM_ATTR onTimer() {
  stepper.tick();
}

void initTimerInterrupt() {
  // 80MHz / 80 / 5 = 200kHz  
  timer = timerBegin(0 /* timer ID */, CPU_FREQ_DIVIDER /* CPU frequency divider */, true /* count up */);
  timerAttachInterrupt(timer, &onTimer, true /* edge */);
  timerAlarmWrite(timer, 5 /* int at counter value */, true /* reload counter */);
  timerAlarmEnable(timer);
}

void log() {
  Serial.printf("%.2f\t%.2f\n", mpu.getAngleDeg(), accel);
}

