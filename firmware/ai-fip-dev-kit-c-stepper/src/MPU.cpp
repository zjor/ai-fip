/**
 * MPU6050 sensor wrapper implementation
 * 
 * @author Sergey Royz (zjor.se@gmail.com)
 * @version 0.1
 */

#include "MPU.h"
#include <Arduino.h>
#include <math.h>

bool MPU::init() {
    Serial.println("Initializing MPU6050...");
    
    if (!mpu.begin()) {
        Serial.println("Failed to initialize MPU6050!");
        return false;
    }
    
    // Configure sensor ranges and filtering
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G); // previous value: MPU6050_RANGE_8_G
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    
    Serial.println("MPU6050 configured successfully");
    
    // Calibrate gyro bias
    Serial.println("Hold still... calibrating gyro bias");
    calibrateGyroRollAxis();
    
    // Initialize Kalman filter with accelerometer reading to avoid startup jump
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    degrees initial_roll = accelRollDegFrom(accel);
    kRoll.setAngle(initial_roll);
    
    // Initialize state variables
    roll_deg = initial_roll;
    accel_roll_deg = initial_roll;
    gyro_roll_deg_per_s = 0.0f;
    
    Serial.printf("MPU6050 initialization complete. Initial roll: %.2f°, Accel bias: %.2f°\n", 
                  initial_roll, accel_roll_bias_deg);
    return true;
}

void MPU::updateAngles(float dt) {
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    
    // Get raw measurements
    accel_roll_deg = accelRollDegFrom(accel);
    
    // Apply bias correction to gyro and convert to deg/s
    radians gyro_roll_rad_per_s = gyroRollRadFrom(gyro) - gyro_roll_bias_rad;
    gyro_roll_deg_per_s = gyro_roll_rad_per_s * RAD_TO_DEG;
    
    // Update Kalman filter
    roll_deg = kRoll.getAngle(accel_roll_deg, gyro_roll_deg_per_s, dt);
    
    // Normalize angle to [-180, 180) degrees
    if (roll_deg >= 180.0f) {
        roll_deg -= 360.0f;
    } else if (roll_deg < -180.0f) {
        roll_deg += 360.0f;
    }
}

void MPU::calibrateGyroRollAxis(size_t samples) {
    sensors_event_t accel, gyro, temp;
    gyro_roll_bias_rad = 0.0f;
    
    // Let sensor settle
    delay(200);
    
    Serial.printf("Collecting %zu samples for gyro calibration...\n", samples);
    
    for (size_t i = 0; i < samples; i++) {
        mpu.getEvent(&accel, &gyro, &temp);
        gyro_roll_bias_rad += gyroRollRadFrom(gyro);
        
        // Show progress every 100 samples
        if (i % 100 == 0) {
            Serial.printf("Calibration progress: %zu/%zu\n", i, samples);
        }
        
        delay(5);  // 5ms between samples
    }
    
    gyro_roll_bias_rad /= samples;
    
    Serial.printf("Gyro calibration complete. Bias: %.6f rad/s (%.3f deg/s)\n", 
                  gyro_roll_bias_rad, gyro_roll_bias_rad * RAD_TO_DEG);
}

degrees MPU::accelRollDegFrom(const sensors_event_t& accel_event) {
    // Calculate roll angle from accelerometer using atan2
    // This assumes the pendulum rotates around the X-axis
    radians roll_rad = atan2f(accel_event.acceleration.y, accel_event.acceleration.z);
    degrees roll_deg = roll_rad * RAD_TO_DEG;
    
    // Apply bias correction
    return roll_deg - accel_roll_bias_deg;
}

radians MPU::gyroRollRadFrom(const sensors_event_t& gyro_event) {
    // Return gyro X-axis reading (roll rate)
    return gyro_event.gyro.x;
}