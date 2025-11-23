#include "../include/display.h"
#include "../include/color.h"
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

// External constants from main.c (needed for frequency labels)
#define MIN_FREQ 27.0f
#define MAX_FREQ 8000.0f

// Color inertia settings
#define HUE_SMOOTHING 0.15f  // 0.0-1.0: lower = smoother/slower, higher = snappier
static float current_hue = 0.0f;
static bool hue_initialized = false;

void visualize_spectrum(float* bands, int num_bands) {
    float max_val = 0.0f;

    for (int i = 0; i < num_bands; i++) {
        float v = bands[i];
        if (!isnan(v) && !isinf(v) && v > max_val)
            max_val = v;
    }
    if (max_val < 1.0f) max_val = 1.0f;

    // Move cursor up 2 lines and clear them
    // (Skip this the first time — rely on static variable)
    static bool first = true;
    if (!first) {
        printf("\033[2A");    // cursor up 2 lines
        printf("\r\033[K");   // clear line
        printf("\n\033[K");   // clear next line
        printf("\r");         // back to start
    }
    first = false;

    // ---------- LINE 1: Spectrum digits ----------
    printf("\r");
    for (int i = 0; i < num_bands; i++) {
        float normalized = bands[i] / max_val;
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 1.0f) normalized = 1.0f;

        int digit = (int)(normalized * 9.0f);
        if(digit > 0){
            printf("\033[1;31m%-4d\033[0m|", digit);
        }else{
            printf("%-4d|", digit);
        }
    }
    printf("\n");

    // ---------- LINE 2: Frequency labels (logarithmic 27Hz - 8kHz) ----------
    printf("\r");
    for (int i = 0; i < num_bands; i++) {
        // Logarithmic frequency scale
        float log_min = logf(MIN_FREQ);
        float log_max = logf(MAX_FREQ);
        float log_freq = log_min + (log_max - log_min) * i / num_bands;
        float freq = expf(log_freq);

        if (freq < 1000.0f) {
            printf("%4.0f|", freq);  // Show Hz for < 1kHz
        } else {
            printf("%3.1fk|", freq / 1000.0f);  // Show kHz for >= 1kHz
        }
    }

    fflush(stdout);
}

void visualize_notes(detected_note_t* notes, int num_notes) {
    printf("\r\033[K"); // Clear line

    // Variables for color output
    uint8_t r = 0, g = 0, b = 0;

    if (num_notes == 0) {
        printf("Notes: (none detected)");
        // Turn off LED when no notes detected
        set_rgb_pwm(0, 0, 0);
    } else {
        printf("Notes: ");

        // Calculate total magnitude for weighted blending
        float total_magnitude = 0.0f;
        for (int i = 0; i < num_notes; i++) {
            total_magnitude += notes[i].magnitude;
        }

        // Circular weighted average of hues using vector math
        // Convert each hue to unit vector, weight by magnitude, sum, then convert back
        float sum_x = 0.0f, sum_y = 0.0f;
        const float deg_to_rad = M_PI / 180.0f;

        for (int i = 0; i < num_notes; i++) {
            // Get note index for color mapping
            int note_idx = get_note_index(notes[i].freq);

            // Print note info
            printf("%s (%.0f Hz)", notes[i].note, notes[i].freq);
            if (i < num_notes - 1) {
                printf(", ");
            }

            // Add weighted hue vector component
            if (note_idx >= 0 && total_magnitude > 0.0f) {
                float hue = get_note_hue(note_idx);
                float weight = notes[i].magnitude / total_magnitude;

                // Convert hue to unit vector and weight
                sum_x += cosf(hue * deg_to_rad) * weight;
                sum_y += sinf(hue * deg_to_rad) * weight;
            }
        }

        // Convert summed vector back to hue angle (this is our target hue)
        float target_hue = atan2f(sum_y, sum_x) * (180.0f / M_PI);
        if (target_hue < 0.0f) {
            target_hue += 360.0f;
        }

        // Apply color inertia using circular interpolation
        if (!hue_initialized) {
            // First time: jump directly to target
            current_hue = target_hue;
            hue_initialized = true;
        } else {
            // Calculate shortest angular distance (handles wraparound)
            float diff = target_hue - current_hue;
            while (diff > 180.0f) diff -= 360.0f;
            while (diff < -180.0f) diff += 360.0f;

            // Apply smoothing
            current_hue += diff * HUE_SMOOTHING;

            // Keep in [0, 360) range
            while (current_hue >= 360.0f) current_hue -= 360.0f;
            while (current_hue < 0.0f) current_hue += 360.0f;
        }

        // Convert to RGB with full saturation and brightness
        hsv_to_rgb(current_hue, 1.0f, 1.0f, &r, &g, &b);

        set_rgb_pwm(r, g, b);

        // Display RGB values
        printf(" [R:%3d G:%3d B:%3d]", r, g, b);
    }

    fflush(stdout);
}
