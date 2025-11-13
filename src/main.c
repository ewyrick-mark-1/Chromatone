#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hardware/adc.h"
#include "hardware/gpio.h"
#include "hardware/dma.h"
#include "pico/stdlib.h"
#include "hardware/timer.h"
#include "hardware/irq.h"

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Configuration
#define FFT_SIZE 1024
#define SAMPLE_RATE 44100
#define OVERLAP_PERCENT 30
#define OVERLAP_SAMPLES ((FFT_SIZE * OVERLAP_PERCENT) / 100)
#define STEP_SIZE (FFT_SIZE - OVERLAP_SAMPLES)  // 717 samples
#define NUM_BANDS 32
#define ADC_CHANNEL 0  // GPIO 26 (not 36 - Pico uses GPIO 26-28 for ADC0-2)

// Buffers
uint16_t adc_buffer[FFT_SIZE * 2];  // Double buffer for DMA
float fft_input[FFT_SIZE];
float fft_real[FFT_SIZE];
float fft_imag[FFT_SIZE];
float magnitude[FFT_SIZE / 2];
float bands[NUM_BANDS];
char display_buffer[4096];  // Buffer for building output before printing

int dma_chan;
volatile bool buffer_ready = false;
volatile int current_write_pos = 0;

// Complex number structure
typedef struct {
    float real;
    float imag;
} Complex;

// FFT Implementation (Cooley-Tukey Radix-2 Decimation-In-Time)
void fft(float* real, float* imag, int n) {
    // Bit reversal
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            // Swap real parts
            float temp = real[i];
            real[i] = real[j];
            real[j] = temp;

            // Swap imaginary parts
            temp = imag[i];
            imag[i] = imag[j];
            imag[j] = temp;
        }

        int k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    // FFT computation
    for (int len = 2; len <= n; len *= 2) {
        float angle = -2.0f * M_PI / len;
        float wlen_real = cosf(angle);
        float wlen_imag = sinf(angle);

        for (int i = 0; i < n; i += len) {
            float w_real = 1.0f;
            float w_imag = 0.0f;

            for (int j = 0; j < len / 2; j++) {
                int idx1 = i + j;
                int idx2 = i + j + len / 2;

                float t_real = w_real * real[idx2] - w_imag * imag[idx2];
                float t_imag = w_real * imag[idx2] + w_imag * real[idx2];

                float u_real = real[idx1];
                float u_imag = imag[idx1];

                real[idx1] = u_real + t_real;
                imag[idx1] = u_imag + t_imag;
                real[idx2] = u_real - t_real;
                imag[idx2] = u_imag - t_imag;

                float w_temp = w_real;
                w_real = w_real * wlen_real - w_imag * wlen_imag;
                w_imag = w_temp * wlen_imag + w_imag * wlen_real;
            }
        }
    }
}

// Hanning window function
void apply_hanning_window(float* data, int n) {
    for (int i = 0; i < n; i++) {
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (n - 1)));
        data[i] *= window;
    }
}

// Calculate magnitude spectrum
void calculate_magnitude(float* real, float* imag, float* mag, int n) {
    for (int i = 0; i < n / 2; i++) {
        mag[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
    }
}

// Map FFT bins to frequency bands (2kHz - 20kHz)
void map_to_bands(float* mag, float* bands, int fft_size, int num_bands) {
    // Frequency per bin
    float freq_per_bin = (float)SAMPLE_RATE / fft_size;

    // Bin indices for 2kHz and 20kHz
    int bin_start = (int)(2000.0f / freq_per_bin);
    int bin_end = (int)(20000.0f / freq_per_bin);

    if (bin_end > fft_size / 2) {
        bin_end = fft_size / 2;
    }

    int bins_per_band = (bin_end - bin_start) / num_bands;
    if (bins_per_band < 1) bins_per_band = 1;

    // Average bins into bands
    for (int i = 0; i < num_bands; i++) {
        int start = bin_start + i * bins_per_band;
        int end = start + bins_per_band;
        if (end > bin_end) end = bin_end;

        float sum = 0.0f;
        int count = 0;
        for (int j = start; j < end; j++) {
            sum += mag[j];
            count++;
        }
        bands[i] = (count > 0) ? (sum / count) : 0.0f;
    }
}

// Visualize spectrum with horizontal bars using buffered output
void visualize_spectrum(float* bands, int num_bands) {
    static bool first_draw = true;
    int pos = 0;  // Position in display_buffer

    // Move cursor to home position (top-left)
    if (first_draw) {
        // First draw: clear entire screen
        pos += sprintf(display_buffer + pos, "\033[2J");
        first_draw = false;
    }
    // Always return cursor to home
    pos += sprintf(display_buffer + pos, "\033[H");

    pos += sprintf(display_buffer + pos, "========================================\n");
    pos += sprintf(display_buffer + pos, "    AUDIO SPECTRUM ANALYZER\n");
    pos += sprintf(display_buffer + pos, "========================================\n");

    // Find max value for normalization
    float max_val = 0.0f;
    for (int i = 0; i < num_bands; i++) {
        if (bands[i] > max_val && !isnan(bands[i]) && !isinf(bands[i])) {
            max_val = bands[i];
        }
    }

    if (max_val < 1.0f) max_val = 1.0f;  // Avoid division by zero

    // Draw bars
    const int bar_width = 50;
    for (int i = 0; i < num_bands; i++) {
        // Sanitize band value
        float band_val = bands[i];
        if (isnan(band_val) || isinf(band_val)) {
            band_val = 0.0f;
        }

        float normalized = band_val / max_val;
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 1.0f) normalized = 1.0f;

        int filled = (int)(normalized * bar_width);

        // Calculate frequency range for this band
        float freq_per_bin = (float)SAMPLE_RATE / FFT_SIZE;
        int bin_start = (int)(2000.0f / freq_per_bin);
        int bin_end = (int)(20000.0f / freq_per_bin);
        int bins_per_band = (bin_end - bin_start) / num_bands;
        float freq_low = (bin_start + i * bins_per_band) * freq_per_bin / 1000.0f;
        float freq_high = (bin_start + (i + 1) * bins_per_band) * freq_per_bin / 1000.0f;

        pos += sprintf(display_buffer + pos, "%5.1f-%5.1fkHz |", freq_low, freq_high);

        for (int j = 0; j < bar_width; j++) {
            if (j < filled) {
                pos += sprintf(display_buffer + pos, "█");
            } else {
                pos += sprintf(display_buffer + pos, "░");
            }
        }

        // Clamp value for display
        int display_val = (int)band_val;
        if (display_val < 0) display_val = 0;
        pos += sprintf(display_buffer + pos, "| %d", display_val);

        // Clear to end of line to erase any leftover characters
        pos += sprintf(display_buffer + pos, "\033[K\n");
    }
    pos += sprintf(display_buffer + pos, "========================================");
    pos += sprintf(display_buffer + pos, "\033[K\n");

    // Now print entire buffer at once
    printf("%s", display_buffer);
    fflush(stdout);  // Force immediate output
}

// DMA interrupt handler
void dma_handler() {
    dma_hw->ints0 = 1u << dma_chan;  // Clear interrupt
    buffer_ready = true;
}

// Setup ADC with DMA
void setup_adc_dma() {
    // Initialize ADC
    adc_init();
    adc_gpio_init(26 + ADC_CHANNEL);  // GPIO 26 = ADC0
    adc_select_input(ADC_CHANNEL);

    // Set up free-running mode
    adc_fifo_setup(
        true,    // Enable FIFO
        true,    // Enable DMA data request
        1,       // Dreq threshold
        false,   // No error bit
        false    // Don't reduce to 8 bits
    );

    // Set sample rate (divider from 48MHz clock)
    // For 44.1kHz: 48MHz / 44.1kHz = 1088
    adc_set_clkdiv(48000000.0f / SAMPLE_RATE);

    // Set up DMA
    dma_chan = dma_claim_unused_channel(true);
    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);

    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_16);
    channel_config_set_read_increment(&cfg, false);  // Always read from ADC FIFO
    channel_config_set_write_increment(&cfg, true);   // Increment write address
    channel_config_set_dreq(&cfg, DREQ_ADC);          // Pace by ADC

    dma_channel_configure(
        dma_chan,
        &cfg,
        adc_buffer,              // Write to buffer
        &adc_hw->fifo,           // Read from ADC FIFO
        FFT_SIZE,                // Transfer count
        false                    // Don't start yet
    );

    // Enable DMA interrupt
    dma_channel_set_irq0_enabled(dma_chan, true);
    irq_set_exclusive_handler(DMA_IRQ_0, dma_handler);
    irq_set_enabled(DMA_IRQ_0, true);
}

// Process audio buffer
void process_audio() {
    // Copy ADC data to float buffer and normalize
    // Remove DC offset by calculating mean first
    float dc_offset = 0.0f;
    for (int i = 0; i < FFT_SIZE; i++) {
        dc_offset += adc_buffer[i];
    }
    dc_offset /= FFT_SIZE;

    // Convert to float and remove DC offset
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_input[i] = (adc_buffer[i] - dc_offset) / 2048.0f;  // 12-bit ADC, remove DC
    }

    // Apply Hanning window
    apply_hanning_window(fft_input, FFT_SIZE);

    // Copy to FFT buffers
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_real[i] = fft_input[i];
        fft_imag[i] = 0.0f;
    }

    // Perform FFT
    fft(fft_real, fft_imag, FFT_SIZE);

    // Calculate magnitude
    calculate_magnitude(fft_real, fft_imag, magnitude, FFT_SIZE);

    // Map to frequency bands
    map_to_bands(magnitude, bands, FFT_SIZE, NUM_BANDS);

    // Visualize
    visualize_spectrum(bands, NUM_BANDS);
}

int main() {
    stdio_init_all();
    sleep_ms(2000);  // Wait for USB serial to connect

    gpio_init(23);
    gpio_set_dir(23, GPIO_OUT);
    gpio_put(23, 1);

    printf("Audio Spectrum Analyzer\n");
    printf("Sample Rate: %d Hz\n", SAMPLE_RATE);
    printf("FFT Size: %d\n", FFT_SIZE);
    printf("Overlap: %d%%\n", OVERLAP_PERCENT);
    printf("Frequency Range: 2-20 kHz\n\n");

    setup_adc_dma();

    // Start ADC
    adc_run(true);

    // Start first DMA transfer
    dma_channel_start(dma_chan);

    int sample_count = 0;

    while(1) {
        if (buffer_ready) {
            buffer_ready = false;

            // Process the audio
            process_audio();

            // Restart DMA with overlap
            // For 30% overlap, we shift by 70% of buffer size
            // This simple implementation restarts full buffer each time
            // A more sophisticated version would use a circular buffer
            dma_channel_set_write_addr(dma_chan, adc_buffer, true);

            sample_count++;
        }

        tight_loop_contents();
    }

    return 0;
}