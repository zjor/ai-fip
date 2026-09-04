/**
 * MPU6050 sensor wrapper with Kalman filtering for inverted pendulum
 * 
 * @author Sergey Royz (zjor.se@gmail.com)
 * @version 0.1
 */

#ifndef MPU_H
#define MPU_H

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Kalman.h>

using radians = float;
using degrees = float;

class MPU {
private:
    Adafruit_MPU6050 mpu;
    Kalman kRoll;
    
    // Bias values
    radians gyro_roll_bias_rad;
    degrees accel_roll_bias_deg;
    
    // Current measurements
    degrees gyro_roll_deg_per_s;  // Angular velocity (deg/s)
    degrees accel_roll_deg;       // Roll from accelerometer
    degrees roll_deg;             // Filtered roll angle
    
    // Helper methods
    void calibrateGyroRollAxis(size_t samples = 800);
    degrees accelRollDegFrom(const sensors_event_t& accel_event);
    radians gyroRollRadFrom(const sensors_event_t& gyro_event);
    
public:
    /**
     * Constructor
     * @param accel_bias Accelerometer roll bias in degrees (default: 0.0)
     */
    explicit MPU(degrees accel_bias = 0.0f) : accel_roll_bias_deg(accel_bias) {};
    /**
     * Initialize the MPU6050 sensor and perform calibration
     * @return true if initialization successful, false otherwise
     */
    bool init();
    
    /**
     * Update all angle measurements using Kalman filter
     * @param dt Time delta in seconds since last update
     */
    void updateAngles(float dt);
    
    /**
     * Get the filtered roll angle in degrees
     * @return Roll angle in degrees [-180, 180)
     */
    degrees getAngleDeg() const { return roll_deg; }
    
    /**
     * Get the angular velocity around roll axis in degrees/second
     * @return Angular velocity in deg/s
     */
    degrees getAngularVelocityDeg() const { return gyro_roll_deg_per_s; }
    
    /**
     * Get the filtered roll angle in radians
     * @return Roll angle in radians [-π, π)
     */
    radians getAngleRad() const { return roll_deg * DEG_TO_RAD; }
    
    /**
     * Get the angular velocity around roll axis in radians/second
     * @return Angular velocity in rad/s
     */
    radians getAngularVelocityRad() const { return gyro_roll_deg_per_s * DEG_TO_RAD; }
    
    /**
     * Get raw accelerometer roll angle (for debugging)
     * @return Raw accelerometer roll in degrees
     */
    degrees getRawAccelRollDeg() const { return accel_roll_deg; }    
};

#endif // MPU_H