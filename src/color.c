#include "../include/color.h"
#include <math.h>
#include "hardware/pwm.h"
#include "hardware/gpio.h"

// Note hue mapping: 12 chromatic notes mapped to color wheel
// Each note is 30 degrees apart (360/12 = 30)
static const float NOTE_HUES[12] = {
    0.0f,    // C  - Red
    30.0f,   // C# - Orange
    60.0f,   // D  - Yellow
    90.0f,   // D# - Yellow-Green
    120.0f,  // E  - Green
    150.0f,  // F  - Cyan-Green
    180.0f,  // F# - Cyan
    210.0f,  // G  - Sky Blue
    240.0f,  // G# - Blue
    270.0f,  // A  - Purple
    300.0f,  // A# - Magenta
    330.0f   // B  - Pink
};

void setup_rgb_pwm(void) {
    // Initialize GPIO pins for PWM function
    gpio_set_function(PWM_R_PIN, GPIO_FUNC_PWM);
    gpio_set_function(PWM_G_PIN, GPIO_FUNC_PWM);
    gpio_set_function(PWM_B_PIN, GPIO_FUNC_PWM);

    // Get PWM slice numbers for each pin
    // GPIO 37 -> Slice 2, Channel B (37/2 = 18, 18%8 = 2)
    // GPIO 38 -> Slice 3, Channel A (38/2 = 19, 19%8 = 3)
    // GPIO 39 -> Slice 3, Channel B (39/2 = 19, 19%8 = 3)
    uint slice_r = pwm_gpio_to_slice_num(PWM_R_PIN);
    uint slice_g = pwm_gpio_to_slice_num(PWM_G_PIN);
    uint slice_b = pwm_gpio_to_slice_num(PWM_B_PIN);

    // Set clock divider to match lab-5-pwm configuration
    pwm_set_clkdiv(slice_r, 150.0f);
    pwm_set_clkdiv(slice_g, 150.0f);
    pwm_set_clkdiv(slice_b, 150.0f);

    // Configure PWM for 8-bit resolution
    pwm_set_wrap(slice_r, PWM_WRAP);
    pwm_set_wrap(slice_g, PWM_WRAP);
    pwm_set_wrap(slice_b, PWM_WRAP);

    // Set initial duty cycle to OFF (common anode: HIGH = off)
    pwm_set_gpio_level(PWM_R_PIN, PWM_WRAP);
    pwm_set_gpio_level(PWM_G_PIN, PWM_WRAP);
    pwm_set_gpio_level(PWM_B_PIN, PWM_WRAP);

    // Enable PWM slices
    pwm_set_enabled(slice_r, true);
    pwm_set_enabled(slice_g, true);
    pwm_set_enabled(slice_b, true);
}

void set_rgb_pwm(uint8_t r, uint8_t g, uint8_t b) {
    // Common anode LED: invert values (HIGH = off, LOW = on)
    pwm_set_gpio_level(PWM_R_PIN, PWM_WRAP - r);
    pwm_set_gpio_level(PWM_G_PIN, PWM_WRAP - g);
    pwm_set_gpio_level(PWM_B_PIN, PWM_WRAP - b);
}

void hsv_to_rgb(float h, float s, float v, uint8_t* r, uint8_t* g, uint8_t* b) {
    // Normalize hue to 0-360
    while (h >= 360.0f) h -= 360.0f;
    while (h < 0.0f) h += 360.0f;

    // Clamp saturation and value
    if (s > 1.0f) s = 1.0f;
    if (s < 0.0f) s = 0.0f;
    if (v > 1.0f) v = 1.0f;
    if (v < 0.0f) v = 0.0f;

    float c = v * s;  // Chroma
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    float r_prime, g_prime, b_prime;

    if (h < 60.0f) {
        r_prime = c; g_prime = x; b_prime = 0;
    } else if (h < 120.0f) {
        r_prime = x; g_prime = c; b_prime = 0;
    } else if (h < 180.0f) {
        r_prime = 0; g_prime = c; b_prime = x;
    } else if (h < 240.0f) {
        r_prime = 0; g_prime = x; b_prime = c;
    } else if (h < 300.0f) {
        r_prime = x; g_prime = 0; b_prime = c;
    } else {
        r_prime = c; g_prime = 0; b_prime = x;
    }

    *r = (uint8_t)((r_prime + m) * 255.0f);
    *g = (uint8_t)((g_prime + m) * 255.0f);
    *b = (uint8_t)((b_prime + m) * 255.0f);
}

int get_note_index(float freq) {
    if (freq < 20.0f) {
        return -1;  // Invalid frequency
    }

    // MIDI formula: n = 12 * log2(f / 440) + 69
    float semitone = 12.0f * log2f(freq / 440.0f) + 69.0f;
    int note_num = (int)(semitone + 0.5f);  // Round to nearest

    if (note_num < 0 || note_num > 127) {
        return -1;
    }

    // Return note index 0-11 (C=0, C#=1, ... B=11)
    return note_num % 12;
}

void note_index_to_rgb(int note_idx, float magnitude, uint8_t* r, uint8_t* g, uint8_t* b) {
    if (note_idx < 0 || note_idx > 11) {
        // Invalid note - turn off LED
        *r = 0;
        *g = 0;
        *b = 0;
        return;
    }

    // Get hue for this note
    float hue = NOTE_HUES[note_idx];

    // Full saturation, brightness based on magnitude
    float saturation = 1.0f;
    float value = magnitude;

    // Clamp value to 0-1 range
    if (value > 1.0f) value = 1.0f;
    if (value < 0.0f) value = 0.0f;

    // Ensure minimum brightness when note is detected
    if (value < 0.2f) value = 0.2f;

    hsv_to_rgb(hue, saturation, value, r, g, b);
}
