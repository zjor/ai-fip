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

static inline float accelRollDegFrom(const sensors_event_t&);
static inline float gyroRollRadFrom(const sensors_event_t&);

hw_timer_t * timer = NULL;

Stepper stepper(PIN_STEPPER_EN, PIN_STEPPER_DIR, PIN_STEPPER_STEP, TICKS_PER_SECOND, PPR, PULSE_WIDTH);

Adafruit_MPU6050 mpu;
Kalman kRoll;

float gyro_bias_rad = 0.0f;
unsigned long last_ms = 0;


inline float normalizeAngle(float value);

void calibrateGyroRollAxis(size_t samples = 400) {
  sensors_event_t a, g, t;
  gyro_bias_rad = 0.0f;
  delay(200); // settle
  for (size_t i = 0; i < samples; i++) {
    mpu.getEvent(&a, &g, &t);
    gyro_bias_rad += gyroRollRadFrom(g);
    delay(2);
  }
  gyro_bias_rad /= samples;
}

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

  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);

  unsigned long now = millis();
  float dt = (now - last_ms) / 1000.0f;
  last_ms = now;

  float roll_accel_deg = accelRollDegFrom(a);
  float gyro_roll_deg_s = (gyroRollRadFrom(g) - gyro_bias_rad) * 180.0f / PI;
  float roll_deg = kRoll.getAngle(roll_accel_deg, gyro_roll_deg_s, dt);

  // Optional: wrap to [-180, 180)
  if (roll_deg >= 180.0f) roll_deg -= 360.0f;
  else if (roll_deg < -180.0f) roll_deg += 360.0f;

  // Output
  Serial.print("Roll (deg): ");
  Serial.println(roll_deg, 2);

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

  Serial.println("Hold still… calibrating gyro bias");
  calibrateGyroRollAxis();

  // Initialize Kalman with accel tilt to avoid startup jump
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  kRoll.setAngle(accelRollDegFrom(a));  // init state

  last_ms = millis();
}


inline float normalizeAngle(float value) {
  return ((value < 180) ? value : value - 360.0f) * DEG_TO_RAD;
}

static inline float accelRollDegFrom(const sensors_event_t& a) {
  float roll_rad = atan2f(a.acceleration.y, a.acceleration.z);
  return roll_rad * RAD_TO_DEG;
}

static inline float gyroRollRadFrom(const sensors_event_t& g) {
  return g.gyro.x;
}