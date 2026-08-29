#include <Arduino.h>
#include <avr/interrupt.h>

#include "Encoder.h"
#include "Motor.h"
#include "PosController.h"

volatile bool sample_flag = false;

void timer1_init_1khz()
{
    cli();
    TCCR1A = 0;
    TCCR1B = 0;

    TCCR1B |= (1 << WGM12); // CTC
    TCCR1B |= (1 << CS11) | (1 << CS10); // prescaler 64

    OCR1A = 249; // 1 kHz

    TIMSK1 |= (1 << OCIE1A);

    sei();
}

ISR(TIMER1_COMPA_vect)
{
    sample_flag = true;
    // pos_update(0.0f);
}

void setup()
{
    Serial.begin(115200);

    motor_init();
    encoder_init();
    delay(1000);
    encoder_reset();

    timer1_init_1khz();
}

void loop()
{
    if (sample_flag) {
        sample_flag = false;
        // pos_update(0.0f);

        // long count = encoder_get();
        // float deg = (float)count * 360.0f / (float)ENC_CPR;

        // Serial.print(count); Serial.print("  "); Serial.print(deg); Serial.print(" | ");
        // Serial.print(error); Serial.print(" "); Serial.println(u);

        motor_set(200);
        delay(1000);
        motor_set(-200);
        delay(1000);
        motor_stop();
        delay(100);
    }

}