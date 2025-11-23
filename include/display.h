#ifndef DISPLAY_H
#define DISPLAY_H

#include <stdint.h>

// Detected note structure
typedef struct {
    float freq;
    float magnitude;  // FFT magnitude of this peak (for weighted color blending)
    char note[5];
} detected_note_t;

// Visualize spectrum as ASCII bar graph
// bands: array of band magnitudes
// num_bands: number of frequency bands
void visualize_spectrum(float* bands, int num_bands);

// Visualize detected notes with RGB color output
// notes: array of detected notes
// num_notes: number of detected notes
// Updates PWM output for dominant note color
void visualize_notes(detected_note_t* notes, int num_notes);

#endif // DISPLAY_H
