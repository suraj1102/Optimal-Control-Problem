#pragma once
#include <Arduino.h>
#include "Encoder.h"
#include "Motor.h"

// tuning gains
#define POS_KP 120.0f
#define POS_KI 0.0f
#define POS_KD 0.0f

// limits
#define POS_INT_MAX 200.0f

// state
static float pos_integral = 0.0f;
static float pos_prev_error = 0.0f;
static unsigned long pos_last_us = 0;
static float u = 0;
static float error = 0;

// wrap angle to [-pi, pi]
static inline float pos_wrap_pi(float x) {
    while (x > PI)  x -= 2.0f * PI;
    while (x < -PI) x += 2.0f * PI;
    return x;
}

// init controller
static inline void pos_init() {
    pos_integral = 0.0f;
    pos_prev_error = 0.0f;
    pos_last_us = micros();
}

// call at fixed rate or as fast as loop runs
// input: desired angle in [-pi, pi]
static inline void pos_update(float target_rad) {
    unsigned long now = micros();
    float dt = (now - pos_last_us) * 1e-6f;
    if (dt <= 0.0f || dt > 0.1f) dt = 0.001f; // clamp
    pos_last_us = now;

    float current = encoder_get_radians();
    error = pos_wrap_pi(current - target_rad);

    // Serial.print("Current (rad): "); Serial.print(current); 
    // Serial.print(" , Error: "); Serial.print(error); Serial.print(" | ");

    // integral
    pos_integral += error * dt;
    if (pos_integral > POS_INT_MAX) pos_integral = POS_INT_MAX;
    if (pos_integral < -POS_INT_MAX) pos_integral = -POS_INT_MAX;

    // derivative
    float derivative = (error - pos_prev_error) / dt;
    pos_prev_error = error;

    // PID
    u = POS_KP * error + POS_KI * pos_integral + POS_KD * derivative;

    motor_set((int)u);
}

// optional stop + reset integrator
static inline void pos_stop() {
    motor_stop();
    pos_integral = 0.0f;
    pos_prev_error = 0.0f;
}