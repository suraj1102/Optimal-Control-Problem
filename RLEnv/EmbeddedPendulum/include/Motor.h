#pragma once
#include <Arduino.h>

// config
#define MOTOR_DEADBAND 60
#define MOTOR_MAX 255

// init pins
static inline void motor_init() {
    pinMode(EN_PIN, OUTPUT);
    pinMode(IN1_PIN, OUTPUT);
    pinMode(IN2_PIN, OUTPUT);

    analogWrite(EN_PIN, 0);
    digitalWrite(IN1_PIN, LOW);
    digitalWrite(IN2_PIN, LOW);
}

// stop motor (brake)
static inline void motor_stop() {
    analogWrite(EN_PIN, 0);
    digitalWrite(IN1_PIN, LOW);
    digitalWrite(IN2_PIN, LOW);
}

// set motor speed and direction
static inline void motor_set(int duty) {
    if (duty < MOTOR_DEADBAND && duty > 0) duty += MOTOR_DEADBAND;
    if (duty > -MOTOR_DEADBAND && duty < 0) duty -= MOTOR_DEADBAND;

    duty = constrain(duty, -MOTOR_MAX, MOTOR_MAX);

    if (duty > 0) {
        digitalWrite(IN1_PIN, HIGH);
        digitalWrite(IN2_PIN, LOW);
        analogWrite(EN_PIN, duty);
    }
    else if (duty < 0) {
        digitalWrite(IN1_PIN, LOW);
        digitalWrite(IN2_PIN, HIGH);
        analogWrite(EN_PIN, -duty); // PWM must be positive
    }
    else {
        motor_stop();
    }
}