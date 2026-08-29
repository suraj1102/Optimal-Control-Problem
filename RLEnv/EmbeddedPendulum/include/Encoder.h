#pragma once
#include <Arduino.h>
#include "PINS.h"

// config
#pragma once
#include <Arduino.h>
#include "PINS.h"

#define ENC_PPR 2500
#define ENC_CPR ENC_PPR   // 1x decoding (important)

volatile long enc_count = 0;

void enc_isr() {
    // fast read
    uint8_t p = PIND;

    uint8_t a = (p >> 3) & 1; // D3
    uint8_t b = (p >> 2) & 1; // D2

    // direction decode
    if (a ^ b) enc_count++;
    else enc_count--;
}

static inline long encoder_get() {
    noInterrupts();
    long v = enc_count;
    interrupts();
    return v;
}

static inline void encoder_reset() {
    noInterrupts();
    enc_count = 0;
    interrupts();
}


// init encoder
static inline void encoder_init() {
    pinMode(ENC_A_PIN, INPUT_PULLUP);
    pinMode(ENC_B_PIN, INPUT_PULLUP);
    pinMode(ENC_Z_PIN, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), enc_isr, CHANGE);
}


static inline float encoder_get_revolutions() {
    return (float)encoder_get() / (float)ENC_CPR;
}

static inline float encoder_get_degrees() {
    return (float)encoder_get() * 360.0f / (float)ENC_CPR;
}

// encoder radians (-inf..inf)
static inline float encoder_get_radians() {
    return (float)encoder_get() * (2.0f * PI / (float)ENC_CPR);
}