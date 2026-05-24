#ifndef COLOR_H
#define COLOR_H

#include <stdint.h>

// PWM pin definitions for RGB LED (matched to physical wiring)
#define PWM_R_PIN 38  // GPIO 37 → Physical RED LED
#define PWM_G_PIN 37  // GPIO 38 → Physical GREEN LED
#define PWM_B_PIN 39  // GPIO 39 → Physical BLUE LED

// PWM resolution (8-bit = 0-255)
#define PWM_WRAP 255

// Initialize PWM hardware for RGB output on pins 37-39
void setup_rgb_pwm(void);

// Set RGB PWM duty cycles (0-255 each)
void set_rgb_pwm(uint8_t r, uint8_t g, uint8_t b);

// Convert HSV to RGB
// h: hue (0-360), s: saturation (0-1), v: value/brightness (0-1)
// r, g, b: output RGB values (0-255)
void hsv_to_rgb(float h, float s, float v, uint8_t* r, uint8_t* g, uint8_t* b);

// Extract note index (0-11) from frequency
// C=0, C#=1, D=2, ... B=11
int get_note_index(float freq);

// Get the hue value (0-360) for a note index (0-11)
float get_note_hue(int note_idx);

// Convert note index to RGB color based on chromatic color wheel
// note_idx: 0-11 (C through B)
// magnitude: used to scale brightness (0.0-1.0 normalized)
// r, g, b: output RGB values (0-255)
void note_index_to_rgb(int note_idx, float magnitude, uint8_t* r, uint8_t* g, uint8_t* b);

#endif // COLOR_H
