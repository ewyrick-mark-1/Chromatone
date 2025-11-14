// refactor_pico_fft.c
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
#define SAMPLE_RATE 44100.0f
#define OVERLAP_PERCENT 30
#define OVERLAP_SAMPLES ((FFT_SIZE * OVERLAP_PERCENT) / 100)
#define STEP_SIZE (FFT_SIZE - OVERLAP_SAMPLES)
#define NUM_BANDS 32
#define ADC_CHANNEL 5      // ADC input index
#define ADC_GPIO_PIN 45    // explicit GPIO pin for ADC channel (use correct board pin)

// Buffers: ping-pong
static uint16_t adc_buffer[2][FFT_SIZE];
static volatile int filled_buffer_index = -1;      // -1 = none, 0 or 1 = newly filled
static volatile bool buffer_ready = false;

// FFT working buffers
static float fft_input[FFT_SIZE];
static float fft_real[FFT_SIZE];
static float fft_imag[FFT_SIZE];
static float magnitude[FFT_SIZE / 2];
static float bands[NUM_BANDS];

static int dma_chan = -1;
static volatile uint32_t dma_interrupt_count = 0;

// ----- forward declarations -----
void fft(float* real, float* imag, int n);
void apply_hanning_window(float* data, int n);
void calculate_magnitude(float* real, float* imag, float* mag, int n);
void map_to_bands(float* mag, float* bands, int fft_size, int num_bands);
void visualize_spectrum(float* bands, int num_bands);
void process_audio_from_buffer(uint16_t *buf);
void setup_adc_dma(void);
void dma_handler(void);

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

void map_to_bands(float* mag, float* out_bands, int fft_size, int num_bands) {
    float freq_per_bin = SAMPLE_RATE / (float)fft_size;
    int bin_start = (int)(2000.0f / freq_per_bin);
    int bin_end = (int)(20000.0f / freq_per_bin);
    if (bin_end > fft_size / 2) bin_end = fft_size / 2;
    int total_bins = bin_end - bin_start;
    int bins_per_band = total_bins / num_bands;
    if (bins_per_band < 1) bins_per_band = 1;

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
        out_bands[i] = (count > 0) ? (sum / (float)count) : 0.0f;
    }
}

void visualize_spectrum(float* bands, int num_bands) {
    float max_val = 0.0f;
    for (int i = 0; i < num_bands; i++) {
        if (bands[i] > max_val && !isnan(bands[i]) && !isinf(bands[i])) max_val = bands[i];
    }
    if (max_val < 1.0f) max_val = 1.0f;

    float freq_per_band = 18000.0f / num_bands;
    for (int i = 0; i < num_bands; i++) {
        float normalized = bands[i] / max_val;
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 1.0f) normalized = 1.0f;
        int digit = (int)(normalized * 9.0f);
        printf("%-4d", digit);
    }
    printf("\n");
    for (int i = 0; i < num_bands; i++) {
        float freq_khz = (2000.0f + i * freq_per_band) / 1000.0f;
        printf("%-4d", (int)freq_khz);
    }
    printf("\n\n");
    fflush(stdout);
}

// ------------------ DMA interrupt handler (ping/pong) ------------------
void dma_handler() {
    // acknowledge/clear interrupt for this channel
    dma_hw->ints0 = 1u << dma_chan;
    dma_interrupt_count++;

    // swap buffer index: figure which buffer was filled
    // The DMA was configured to write FFT_SIZE samples, starting at whichever address we gave it.
    // We track the write address externally via current buffer index; here we'll toggle.
    static int next = 0;
    filled_buffer_index = next;   // indicate which buffer is filled
    next ^= 1;                    // toggle for next time
    buffer_ready = true;
}

// ------------------ ADC + DMA setup ------------------
void setup_adc_dma(void) {
    adc_init();
    adc_gpio_init(ADC_GPIO_PIN);
    adc_select_input(ADC_CHANNEL);

    adc_fifo_setup(
        true,   // enable FIFO
        true,   // enable DMA request (DREQ)
        1,      // DREQ when >=1 sample in FIFO
        false,  // don't set ERR bit
        false   // no 8-bit conversion
    );

    // set requested sample rate (SDK may clamp if impossible)
    adc_set_clkdiv(48000000.0f / SAMPLE_RATE);

    // DMA config
    dma_chan = dma_claim_unused_channel(true);
    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);
    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_16);
    channel_config_set_read_increment(&cfg, false);  // read from ADC FIFO (no increment)
    channel_config_set_write_increment(&cfg, true);  // write sequentially into buffer
    channel_config_set_dreq(&cfg, DREQ_ADC);

    // Initially point to buffer 0
    dma_channel_configure(
        dma_chan,
        &cfg,
        adc_buffer[0],         // write address
        &adc_hw->fifo,         // read from ADC FIFO
        FFT_SIZE,              // transfer count
        false                  // don't start yet
    );

    // IRQ setup
    dma_channel_set_irq0_enabled(dma_chan, true);
    irq_set_exclusive_handler(DMA_IRQ_0, dma_handler);
    irq_set_enabled(DMA_IRQ_0, true);
}

// ------------------ Audio processing for one buffer ------------------
void process_audio_from_buffer(uint16_t *buf) {
    // compute DC offset
    float dc = 0.0f;
    for (int i = 0; i < FFT_SIZE; i++) dc += buf[i];
    dc /= (float)FFT_SIZE;

    // convert to floats and normalize to approx ±1.0 (12-bit)
    float energy = 0.0f;
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_input[i] = ((float)buf[i] - dc) / 2048.0f;
        energy += fabsf(fft_input[i]);
    }
    float avg_abs = energy / (float)FFT_SIZE;

    // debug: print RMS/mean
    float sum_sq = 0.0f;
    for (int i = 0; i < FFT_SIZE; i++) sum_sq += fft_input[i] * fft_input[i];
    float rms = sqrtf(sum_sq / (float)FFT_SIZE);
    printf("RMS=%.6f avg=%.6f\n", rms, avg_abs);

    // window and FFT
    apply_hanning_window(fft_input, FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_real[i] = fft_input[i];
        fft_imag[i] = 0.0f;
    }
    fft(fft_real, fft_imag, FFT_SIZE);
    calculate_magnitude(fft_real, fft_imag, magnitude, FFT_SIZE);

    // find peak bin (debug)
    float max_val = 0.0f;
    int max_bin = 0;
    for (int i = 0; i < FFT_SIZE/2; i++) {
        if (magnitude[i] > max_val) {
            max_val = magnitude[i];
            max_bin = i;
        }
    }
    float peak_freq = (float)max_bin * SAMPLE_RATE / (float)FFT_SIZE;
    printf("Peak bin=%d freq=%.1fHz mag=%.6f\n", max_bin, peak_freq, max_val);

    // map and visualize
    map_to_bands(magnitude, bands, FFT_SIZE, NUM_BANDS);
    visualize_spectrum(bands, NUM_BANDS);
}

// ------------------ Main ------------------
int main() {
    stdio_init_all();
    sleep_ms(1000);

    // LED heartbeat
    gpio_init(23);
    gpio_set_dir(23, GPIO_OUT);
    gpio_put(23, 1);

    printf("\n=== BOOT ===\n");

    setup_adc_dma();

    // Start ADC and begin DMA into buffer 0
    adc_run(true);
    dma_channel_start(dma_chan);

    printf("Waiting for first DMA fill...\n");

    while (1) {
        if (buffer_ready) {
            // take ownership of the filled buffer
            int bufid = filled_buffer_index;
            buffer_ready = false;
            filled_buffer_index = -1;

            // process it
            process_audio_from_buffer(adc_buffer[bufid]);

            // drain ADC FIFO before we reprogram DMA write address to avoid stale samples
            adc_fifo_drain();

            // reconfigure DMA to the other buffer for the next transfer
            int next = bufid ^ 1;
            dma_channel_set_write_addr(dma_chan, adc_buffer[next], true);

            // continue loop; DMA will fill the "next" buffer and interrupt when done
        }

        tight_loop_contents();
    }

    return 0;
}
