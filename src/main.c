// Single-channel DMA-based FFT for instrument note detection
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hardware/adc.h"
#include "hardware/gpio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "pico/stdlib.h"

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Configuration
#define FFT_SIZE 16384
#define NUM_BANDS 32
#define OVERLAP_PERCENT 50
#define STEP_SIZE (FFT_SIZE * (100 - OVERLAP_PERCENT) / 100)  // 50% overlap = 8192 step
#define NOISE_FLOOR_THRESHOLD 3.0f  // Ignore FFT magnitudes below this value

// Single-channel configuration for instrument range
#define SAMPLE_RATE 20000.0f
#define ADC_CHANNEL 0
#define GPIO_PIN 40
#define MIN_FREQ 27.0f   // A0 fundamental
#define MAX_FREQ 8000.0f // Upper instrument range

// DMA ping-pong buffers (fill STEP_SIZE at a time for 50% overlap)
static uint16_t adc_buffer[2][STEP_SIZE];
static volatile int filled_buffer_index = -1;
static volatile bool buffer_ready = false;

// Overlap buffer for 50% overlap
static uint16_t overlap_buffer[STEP_SIZE];

// DMA channels
static int dma_chan_0;
static int dma_chan_1;

// Debug counter
static volatile uint32_t fft_count = 0;

// Display mode toggle (0=spectrum, 1=notes) - change this to switch modes
static volatile bool display_mode = 1;

// FFT working buffers
static float fft_input[FFT_SIZE];
static float fft_real[FFT_SIZE];
static float fft_imag[FFT_SIZE];
static float magnitude[FFT_SIZE / 2];
static float spectrum[NUM_BANDS];

// Detected note structure
typedef struct {
    float freq;
    char note[5];
} detected_note_t;

// ----- forward declarations -----
void fft(float* real, float* imag, int n);
void apply_hanning_window(float* data, int n);
void calculate_magnitude(float* real, float* imag, float* mag, int n);
void map_to_bands(float* mag, float* bands, int fft_size, int num_bands,
                  float sample_rate, float min_freq, float max_freq);
void freq_to_note(float freq, char* note_str);
int find_peaks(detected_note_t* notes, int max_notes);
void visualize_spectrum(float* bands, int num_bands);
void visualize_notes(detected_note_t* notes, int num_notes);
void dma_handler(void);
void setup_adc_dma(void);
void process_fft_buffer(uint16_t* buffer);

// ------------------ FFT (kept your implementation) ------------------
void fft(float* real, float* imag, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float tmp = real[i]; real[i] = real[j]; real[j] = tmp;
            tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
        }
        int k = n / 2;
        while (k <= j) { j -= k; k /= 2; }
        j += k;
    }

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

// ------------------ Window / mag / mapping / viz ------------------
void apply_hanning_window(float* data, int n) {
    for (int i = 0; i < n; i++) {
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (n - 1)));
        data[i] *= window;
    }
}

void calculate_magnitude(float* real, float* imag, float* mag, int n) {
    for (int i = 0; i < n / 2; i++) {
        mag[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i]);
    }
}

// Map FFT magnitude to logarithmic bands
void map_to_bands(float* mag, float* bands, int fft_size, int num_bands,
                  float sample_rate, float min_freq, float max_freq) {
    // Clear bands first
    for (int i = 0; i < num_bands; i++) {
        bands[i] = 0.0f;
    }

    float freq_per_bin = sample_rate / (float)fft_size;

    // Calculate bin range
    int bin_start = (int)(min_freq / freq_per_bin);
    int bin_end = (int)(max_freq / freq_per_bin);

    if (bin_end > fft_size / 2)
        bin_end = fft_size / 2;
    if (bin_start < 0)
        bin_start = 0;

    // Map to logarithmic frequency bands
    float log_min = logf(min_freq);
    float log_max = logf(max_freq);

    for (int bin = bin_start; bin < bin_end; bin++) {
        float bin_freq = bin * freq_per_bin;

        if (bin_freq < min_freq || bin_freq > max_freq)
            continue;

        float log_freq = logf(bin_freq);

        // Map to band index
        int band_idx = (int)((log_freq - log_min) / (log_max - log_min) * num_bands);

        if (band_idx >= 0 && band_idx < num_bands) {
            // Take maximum value in each band
            if (mag[bin] > bands[band_idx]) {
                bands[band_idx] = mag[bin];
            }
        }
    }
}


// Convert frequency to note name (A0-C8)
void freq_to_note(float freq, char* note_str) {
    if (freq < 20.0f) {
        strcpy(note_str, "---");
        return;
    }

    // Note names
    const char* notes[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

    // A4 = 440 Hz is note number 69 in MIDI (semitone 69 from C0)
    // Formula: n = 12 * log2(f / 440) + 69
    float semitone = 12.0f * log2f(freq / 440.0f) + 69.0f;
    int note_num = (int)(semitone + 0.5f); // Round to nearest semitone

    if (note_num < 0 || note_num > 127) {
        strcpy(note_str, "---");
        return;
    }

    int octave = (note_num / 12) - 1;
    int note_idx = note_num % 12;

    sprintf(note_str, "%s%d", notes[note_idx], octave);
}

// Find peaks in FFT magnitude array and convert to notes
int find_peaks(detected_note_t* notes, int max_notes) {
    int num_notes = 0;
    float threshold = 50.0f; // Minimum magnitude to consider as peak
    float freq_per_bin = SAMPLE_RATE / (float)FFT_SIZE;

    // Scan through FFT bins looking for local maxima
    for (int i = 10; i < FFT_SIZE / 2 - 10 && num_notes < max_notes; i++) {
        // Check if this is a local maximum
        if (magnitude[i] > threshold &&
            magnitude[i] > magnitude[i-1] &&
            magnitude[i] > magnitude[i+1] &&
            magnitude[i] > magnitude[i-2] &&
            magnitude[i] > magnitude[i+2]) {

            // Calculate frequency
            float freq = i * freq_per_bin;

            // Only include frequencies in instrument range
            if (freq >= MIN_FREQ && freq <= MAX_FREQ) {
                notes[num_notes].freq = freq;
                freq_to_note(freq, notes[num_notes].note);
                num_notes++;
            }
        }
    }

    return num_notes;
}

// Display detected notes
void visualize_notes(detected_note_t* notes, int num_notes) {
    printf("\r\033[K"); // Clear line
    printf("Notes in signal: ");

    if (num_notes == 0) {
        printf("(none detected)");
    } else {
        for (int i = 0; i < num_notes; i++) {
            printf("%s (%.0f Hz)", notes[i].note, notes[i].freq);
            if (i < num_notes - 1) {
                printf(", ");
            }
        }
    }

    fflush(stdout);
}

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


// ------------------ DMA interrupt handler ------------------
void dma_handler(void) {
    // Check which DMA channel completed
    if (dma_hw->ints0 & (1u << dma_chan_0)) {
        dma_hw->ints0 = 1u << dma_chan_0;
        filled_buffer_index = 0;
        buffer_ready = true;
        gpio_put(24, 0);  // Toggle LED for debug

        // Reset channel 0 for next time it gets chained to
        dma_channel_set_write_addr(dma_chan_0, adc_buffer[0], false);
        dma_channel_set_trans_count(dma_chan_0, STEP_SIZE, false);
    }
    if (dma_hw->ints0 & (1u << dma_chan_1)) {
        dma_hw->ints0 = 1u << dma_chan_1;
        filled_buffer_index = 1;
        buffer_ready = true;
        gpio_put(24, 1);  // Toggle LED for debug

        // Reset channel 1 for next time it gets chained to
        dma_channel_set_write_addr(dma_chan_1, adc_buffer[1], false);
        dma_channel_set_trans_count(dma_chan_1, STEP_SIZE, false);
    }
}

// ------------------ ADC + DMA setup ------------------
void setup_adc_dma(void) {
    // Initialize ADC
    adc_init();
    adc_gpio_init(GPIO_PIN);
    adc_select_input(ADC_CHANNEL);

    // Configure ADC for free-running at 20 kHz
    // ADC clock = 48 MHz, divider = 48000000 / 20000 = 2400
    adc_set_clkdiv(2400.0f - 1.0f);

    // Setup FIFO for DMA
    adc_fifo_setup(
        true,    // Write to FIFO
        true,    // Enable DMA requests
        1,       // DREQ threshold
        false,   // No error bit
        false    // 12-bit samples (not 8-bit)
    );

    // Claim two DMA channels for ping-pong
    dma_chan_0 = dma_claim_unused_channel(true);
    dma_chan_1 = dma_claim_unused_channel(true);

    // Configure DMA channel 0
    dma_channel_config c0 = dma_channel_get_default_config(dma_chan_0);
    channel_config_set_transfer_data_size(&c0, DMA_SIZE_16);
    channel_config_set_read_increment(&c0, false);
    channel_config_set_write_increment(&c0, true);
    channel_config_set_dreq(&c0, DREQ_ADC);
    channel_config_set_chain_to(&c0, dma_chan_1);

    dma_channel_configure(
        dma_chan_0,
        &c0,
        adc_buffer[0],      // Write to buffer 0
        &adc_hw->fifo,      // Read from ADC FIFO
        STEP_SIZE,          // Transfer count (8192 for 50% overlap)
        false               // Don't start yet
    );

    // Configure DMA channel 1
    dma_channel_config c1 = dma_channel_get_default_config(dma_chan_1);
    channel_config_set_transfer_data_size(&c1, DMA_SIZE_16);
    channel_config_set_read_increment(&c1, false);
    channel_config_set_write_increment(&c1, true);
    channel_config_set_dreq(&c1, DREQ_ADC);
    channel_config_set_chain_to(&c1, dma_chan_0);

    dma_channel_configure(
        dma_chan_1,
        &c1,
        adc_buffer[1],      // Write to buffer 1
        &adc_hw->fifo,      // Read from ADC FIFO
        STEP_SIZE,          // Transfer count (8192 for 50% overlap)
        false               // Don't start yet
    );

    // Enable DMA interrupts
    dma_channel_set_irq0_enabled(dma_chan_0, true);
    dma_channel_set_irq0_enabled(dma_chan_1, true);

    irq_set_exclusive_handler(DMA_IRQ_0, dma_handler);
    irq_set_enabled(DMA_IRQ_0, true);

    // Start ADC and first DMA channel
    adc_run(true);
    dma_channel_start(dma_chan_0);
}

// Process FFT with 50% overlap
void process_fft_buffer(uint16_t* new_buffer) {
    fft_count++;

    // Combine overlap buffer (first 8192) with new data (next 8192)
    // First copy overlap buffer to fft_input
    for (int i = 0; i < STEP_SIZE; i++) {
        fft_input[i] = (float)overlap_buffer[i];
    }

    // Then copy all of new buffer (which is STEP_SIZE samples)
    for (int i = 0; i < STEP_SIZE; i++) {
        fft_input[STEP_SIZE + i] = (float)new_buffer[i];
    }

    // Save new buffer for next overlap
    memcpy(overlap_buffer, new_buffer, STEP_SIZE * sizeof(uint16_t));

    // Compute DC offset
    float dc = 0.0f;
    for (int i = 0; i < FFT_SIZE; i++) {
        dc += fft_input[i];
    }
    dc /= (float)FFT_SIZE;

    // Normalize to ±1.0 range (12-bit ADC)
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_input[i] = (fft_input[i] - dc) / 2048.0f;
    }

    // Apply Hanning window
    apply_hanning_window(fft_input, FFT_SIZE);

    // Prepare for FFT
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_real[i] = fft_input[i];
        fft_imag[i] = 0.0f;
    }

    // Perform FFT
    fft(fft_real, fft_imag, FFT_SIZE);
    calculate_magnitude(fft_real, fft_imag, magnitude, FFT_SIZE);

    // Apply noise floor threshold
    for (int i = 0; i < FFT_SIZE / 2; i++) {
        if (magnitude[i] < NOISE_FLOOR_THRESHOLD) {
            magnitude[i] = 0.0f;
        }
    }

    // Map to logarithmic bands
    map_to_bands(magnitude, spectrum, FFT_SIZE, NUM_BANDS,
                 SAMPLE_RATE, MIN_FREQ, MAX_FREQ);

    // Print debug info every 10 FFTs (only in spectrum mode)
    if (fft_count % 10 == 0 && display_mode == 0) {
        printf("[FFT #%lu] ", fft_count);
    }

    // Visualize based on display mode
    if (display_mode == 0) {
        // Mode 0: Show spectrum
        visualize_spectrum(spectrum, NUM_BANDS);
    } else {
        // Mode 1: Show detected notes
        detected_note_t notes[20]; // Max 20 notes
        int num_notes = find_peaks(notes, 20);
        visualize_notes(notes, num_notes);
    }
}

// ------------------ Main ------------------
int main() {
    // Increase voltage for stable overclocking
    //vreg_set_voltage(VREG_VOLTAGE_1_20);
    //sleep_ms(10);

    // Overclock to 300 MHz for faster FFT processing
    set_sys_clock_khz(300000, true);

    stdio_init_all();
    sleep_ms(1000);

    // LED heartbeat
    gpio_init(24);
    gpio_set_dir(24, GPIO_OUT);
    gpio_put(24, 1);

    uint32_t actual_freq_khz = clock_get_hz(clk_sys) / 1000;

    printf("\n=== Single-Channel DMA FFT: Instrument Note Detection ===\n");
    printf("CPU Clock: %lu MHz (requested 300 MHz)\n", actual_freq_khz / 1000);
    printf("Change display_mode variable (line 46) to toggle: 0=Spectrum, 1=Note Detection\n");
    printf("Range: %.1f Hz - %.1f kHz (A0 to upper instrument range)\n",
           MIN_FREQ, MAX_FREQ / 1000.0f);
    printf("Sample rate: %.1f kHz, FFT Size: %d\n",
           SAMPLE_RATE / 1000.0f, FFT_SIZE);
    printf("Frequency resolution: %.2f Hz\n", SAMPLE_RATE / FFT_SIZE);
    printf("Overlap: %d%%, Update rate: ~%.1f Hz\n\n",
           OVERLAP_PERCENT, SAMPLE_RATE / STEP_SIZE);

    // Initialize overlap buffer with zeros
    memset(overlap_buffer, 0, STEP_SIZE * sizeof(uint16_t));

    // Initialize spectrum
    for (int i = 0; i < NUM_BANDS; i++) {
        spectrum[i] = 0.0f;
    }

    // Setup ADC and DMA
    setup_adc_dma();

    printf("Waiting for initial buffer fill...\n");
    sleep_ms(1000);

    printf("Starting FFT processing...\n\n");

    // Main loop: wait for DMA buffers and process
    while (1) {
        // Process FFT if buffer ready
        if (buffer_ready) {
            buffer_ready = false;

            // Process the filled buffer
            process_fft_buffer(adc_buffer[filled_buffer_index]);
        }

        tight_loop_contents();
    }

    return 0;
}