/**
 * AI-controlled flywheel inverted pendulum firmware.
 * 
 * Parts:
 *  - ESP32 DevKit-C
 *  - Nema17 stepper motor
 *  - A4988 stepper driver
 *  - MPU9250 
 * 
 * @author Sergey Royz (zjor.se@gmail.com) 
 * @version 0.1
 * @date 2025-05-17 - ...
 */
#include <Arduino.h>
#include <Wire.h>
#include <SparkFunMPU9250-DMP.h>

#include <math.h>

#include "pinout.h"
#include "stepper/stepper.h"

// pulses per revolution
#define PPR               1600

#define CPU_FREQ_MHZ      240
#define CPU_FREQ_DIVIDER  120
#define TICKS_PER_SECOND  (200000 * (CPU_FREQ_MHZ / CPU_FREQ_DIVIDER))
#define PULSE_WIDTH       1

void initTimerInterrupt();
void updateVelocity(unsigned long);
void log(unsigned long);

hw_timer_t * timer = NULL;

Stepper stepper(PIN_STEPPER_EN, PIN_STEPPER_DIR, PIN_STEPPER_STEP, TICKS_PER_SECOND, PPR, PULSE_WIDTH);

MPU9250_DMP imu;
volatile bool dmpDataReady = false;
float roll, pitch, yaw;

void dmpISR();
void initIMU();
bool readIMU();


void setup() {
  setCpuFrequencyMhz(CPU_FREQ_MHZ);

  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(1000000UL);

  // initTimerInterrupt();
  delay(500);
  initIMU();

  // stepper.init();
  // stepper.setEnabled(true);


}

void loop() {
  // unsigned long nowMicros = micros();
  // updateVelocity(nowMicros);
  // readIMU();
  // log(nowMicros);
}

float t = 0.0;
void updateVelocity(unsigned long nowMicros) {
  static unsigned long timestamp = micros();
  if (nowMicros - timestamp < 50 /* 20 kHz */) {
    return;
  }
  
  float dt = ((float) (nowMicros - timestamp)) * 1e-6;
  if (dt < 1.0) {
    t += dt;
  }
  float v = 12.0 * sin(2 * M_PI / 0.96 * t);
  stepper.setVelocity(v);
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
  if (nowMicros - timestamp < 100000 /* 10 Hz */) {
    return;
  }

  Serial.printf("%.2f\t%.2f\t%.2f\n", roll, pitch, yaw);
}

void dmpISR() {
  dmpDataReady = true;
}

void initIMU() {
  pinMode(PIN_IMU_INT, INPUT_PULLUP);

  Serial.println("before MPU begin");
  if (imu.begin() != INV_SUCCESS) {
    Serial.println("after MPU begin");
    while (1) {
      Serial.println("Failed to init IMU");
      delay(5000);
    }
  }

  Serial.println("MPU ready");

  imu.dmpBegin(DMP_FEATURE_6X_LP_QUAT | DMP_FEATURE_GYRO_CAL, 100);
  imu.enableInterrupt();
  imu.setIntLevel(INT_ACTIVE_LOW);
  imu.setIntLatched(INT_LATCHED);  

  attachInterrupt(PIN_IMU_INT, dmpISR, FALLING);
}

bool readIMU() {
  if (dmpDataReady && imu.fifoAvailable()) {
    if (imu.dmpUpdateFifo() == INV_SUCCESS) {
      dmpDataReady = false;
      imu.computeEulerAngles();
      roll = imu.roll;
      pitch = imu.pitch;
      yaw = imu.yaw;
      return true;
    }
  }
  return false;
}
