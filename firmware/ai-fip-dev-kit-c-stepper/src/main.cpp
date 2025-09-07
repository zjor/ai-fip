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

#include <math.h>

#include "pinout.h"
#include "stepper/stepper.h"

// pulses per revolution
#define PPR               1600

#define CPU_FREQ_MHZ      240
#define CPU_FREQ_DIVIDER  120
#define TICKS_PER_SECOND  (200000 * (CPU_FREQ_MHZ / CPU_FREQ_DIVIDER))
#define PULSE_WIDTH       1

void initMPU();
void initTimerInterrupt();
void updateVelocity(unsigned long);
void updateControl(unsigned long);
void log(unsigned long);

hw_timer_t * timer = NULL;

Stepper stepper(PIN_STEPPER_EN, PIN_STEPPER_DIR, PIN_STEPPER_STEP, TICKS_PER_SECOND, PPR, PULSE_WIDTH);

Adafruit_MPU6050 mpu;

inline float normalizeAngle(float value);

void setup() {
  setCpuFrequencyMhz(CPU_FREQ_MHZ);

  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(1000000UL);

  initTimerInterrupt();
  delay(500);
  initMPU();

  stepper.init();
  stepper.setEnabled(true);
}

void loop() {
  unsigned long nowMicros = micros();
  updateVelocity(nowMicros);
  updateControl(nowMicros);

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  Serial.print("Accel (m/s^2): ");
  Serial.print(a.acceleration.x);
  Serial.print(", ");
  Serial.print(a.acceleration.y);
  Serial.print(", ");
  Serial.println(a.acceleration.z);

  Serial.print("Gyro (rad/s): ");
  Serial.print(g.gyro.x);
  Serial.print(", ");
  Serial.print(g.gyro.y);
  Serial.print(", ");
  Serial.println(g.gyro.z);

  Serial.print("Temperature (°C): ");
  Serial.println(temp.temperature);

  Serial.println("----");
  delay(500);  
}

void updateVelocity(unsigned long nowMicros) {
  static unsigned long timestamp = micros();
  if (nowMicros - timestamp < 50 /* 20 kHz */) {
    return;
  }
  
  float dt = ((float) (nowMicros - timestamp)) * 1e-6;
  // velocity += accel * dt;
  // velocity = angle * 10.0;
  // stepper.setVelocity(velocity);
  timestamp = nowMicros;
}

void updateControl(unsigned long nowMicros) {
  static unsigned long timestamp = micros();
  if (nowMicros - timestamp < 1000 /* 1kHz*/) {
    return;
  }
  // angle = normalizeAngle(roll);
  // if (abs(angle) < 0.5) {
  //   accel += angle / 100.0;
  // }
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

void log(unsigned long nowMicros) {
  static unsigned long timestamp = micros();
  if (nowMicros - timestamp < 1000000 /* 10 Hz */) {
    return;
  }

  // Serial.printf("%.2f\t%.2f\t%.2f\t%.4f\n", roll, pitch, yaw, angle);
}


void initMPU() {
  Serial.println("before MPU begin");
  if (!mpu.begin()) {
    Serial.println("after MPU begin");
    while (1) {
      Serial.println("Failed to init IMU");
      delay(500);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("MPU ready");
}


inline float normalizeAngle(float value) {
  return ((value < 180) ? value : value - 360.0f) * DEG_TO_RAD;
}