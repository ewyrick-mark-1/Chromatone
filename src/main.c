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
#define NUM_BANDS 32

// Three-channel configuration
#define NUM_CHANNELS 3

// Channel 0: Low frequency (20 - 7447 Hz)
#define CH0_SAMPLE_RATE 16000.0f
#define CH0_ADC_CHANNEL 0
#define CH0_GPIO_PIN 26
#define CH0_MIN_FREQ 20.0f
#define CH0_MAX_FREQ 7447.0f

// Channel 1: Mid frequency (7242 - 14773 Hz)
#define CH1_SAMPLE_RATE 32000.0f
#define CH1_ADC_CHANNEL 1
#define CH1_GPIO_PIN 27
#define CH1_MIN_FREQ 7242.0f
#define CH1_MAX_FREQ 14773.0f

// Channel 2: High frequency (14573 - 22000 Hz)
#define CH2_SAMPLE_RATE 48000.0f
#define CH2_ADC_CHANNEL 2
#define CH2_GPIO_PIN 28
#define CH2_MIN_FREQ 14573.0f
#define CH2_MAX_FREQ 22000.0f

// Per-channel data structure
typedef struct {
    uint16_t adc_buffer[2][FFT_SIZE];     // ping-pong buffers
    volatile int filled_buffer_index;      // -1 = none, 0 or 1 = newly filled
    volatile bool buffer_ready;
    int dma_chan;
    float sample_rate;
    int adc_channel;
    int gpio_pin;
    float min_freq;
    float max_freq;
    volatile uint32_t dma_interrupt_count;
} channel_t;

static channel_t channels[NUM_CHANNELS];

// FFT working buffers (shared across channels)
static float fft_input[FFT_SIZE];
static float fft_real[FFT_SIZE];
static float fft_imag[FFT_SIZE];
static float magnitude[FFT_SIZE / 2];
static float combined_spectrum[NUM_BANDS];  // Combined output from all channels

// ----- forward declarations -----
void fft(float* real, float* imag, int n);
void apply_hanning_window(float* data, int n);
void calculate_magnitude(float* real, float* imag, float* mag, int n);
void map_channel_to_bands(float* mag, float* bands, int fft_size, int num_bands,
                          float sample_rate, float min_freq, float max_freq);
void visualize_spectrum(float* bands, int num_bands);
void process_audio_from_buffer(uint16_t *buf, int ch_idx);
void setup_adc_dma(void);
void start_channel_sampling(int ch_idx);
void dma_handler_ch0(void);
void dma_handler_ch1(void);
void dma_handler_ch2(void);

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

// Map FFT magnitude to bands for a specific frequency range
// This version contributes to the combined spectrum based on log-scale frequency mapping
void map_channel_to_bands(float* mag, float* bands, int fft_size, int num_bands,
                          float sample_rate, float min_freq, float max_freq) {
    float freq_per_bin = sample_rate / (float)fft_size;

    // Calculate bin range for this channel
    int bin_start = (int)(min_freq / freq_per_bin);
    int bin_end = (int)(max_freq / freq_per_bin);

    if (bin_end > fft_size / 2)
        bin_end = fft_size / 2;
    if (bin_start < 0)
        bin_start = 0;

    // Map to logarithmic frequency bands across full spectrum (20Hz - 20kHz)
    // We'll map each FFT bin to the appropriate band in the output
    for (int bin = bin_start; bin < bin_end; bin++) {
        float bin_freq = bin * freq_per_bin;

        // Calculate which band this frequency belongs to (log scale)
        // Band mapping: logarithmic from 20Hz to 20kHz
        float log_min = logf(20.0f);
        float log_max = logf(20000.0f);
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

    // ---------- LINE 2: Frequency labels (logarithmic 20Hz - 20kHz) ----------
    printf("\r");
    for (int i = 0; i < num_bands; i++) {
        // Logarithmic frequency scale
        float log_min = logf(20.0f);
        float log_max = logf(20000.0f);
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


// ------------------ DMA interrupt handlers (one per channel) ------------------
void dma_handler_ch0() {
    channel_t *ch = &channels[0];
    dma_hw->ints0 = 1u << ch->dma_chan;
    ch->dma_interrupt_count++;

    static int next = 0;
    ch->filled_buffer_index = next;
    next ^= 1;
    ch->buffer_ready = true;
}

void dma_handler_ch1() {
    channel_t *ch = &channels[1];
    dma_hw->ints0 = 1u << ch->dma_chan;
    ch->dma_interrupt_count++;

    static int next = 0;
    ch->filled_buffer_index = next;
    next ^= 1;
    ch->buffer_ready = true;
}

void dma_handler_ch2() {
    channel_t *ch = &channels[2];
    dma_hw->ints0 = 1u << ch->dma_chan;
    ch->dma_interrupt_count++;

    static int next = 0;
    ch->filled_buffer_index = next;
    next ^= 1;
    ch->buffer_ready = true;
}

// ------------------ ADC + DMA setup for 3 channels ------------------
// Uses sequential time-division: each channel gets sampled independently
void setup_adc_dma(void) {
    adc_init();

    // Initialize channel configurations
    channels[0].sample_rate = CH0_SAMPLE_RATE;
    channels[0].adc_channel = CH0_ADC_CHANNEL;
    channels[0].gpio_pin = CH0_GPIO_PIN;
    channels[0].min_freq = CH0_MIN_FREQ;
    channels[0].max_freq = CH0_MAX_FREQ;
    channels[0].filled_buffer_index = -1;
    channels[0].buffer_ready = false;
    channels[0].dma_interrupt_count = 0;

    channels[1].sample_rate = CH1_SAMPLE_RATE;
    channels[1].adc_channel = CH1_ADC_CHANNEL;
    channels[1].gpio_pin = CH1_GPIO_PIN;
    channels[1].min_freq = CH1_MIN_FREQ;
    channels[1].max_freq = CH1_MAX_FREQ;
    channels[1].filled_buffer_index = -1;
    channels[1].buffer_ready = false;
    channels[1].dma_interrupt_count = 0;

    channels[2].sample_rate = CH2_SAMPLE_RATE;
    channels[2].adc_channel = CH2_ADC_CHANNEL;
    channels[2].gpio_pin = CH2_GPIO_PIN;
    channels[2].min_freq = CH2_MIN_FREQ;
    channels[2].max_freq = CH2_MAX_FREQ;
    channels[2].filled_buffer_index = -1;
    channels[2].buffer_ready = false;
    channels[2].dma_interrupt_count = 0;

    // Initialize GPIO pins for ADC
    for (int i = 0; i < NUM_CHANNELS; i++) {
        adc_gpio_init(channels[i].gpio_pin);
    }

    // Claim DMA channels
    channels[0].dma_chan = dma_claim_unused_channel(true);
    channels[1].dma_chan = dma_claim_unused_channel(true);
    channels[2].dma_chan = dma_claim_unused_channel(true);

    // Configure FIFO (will be reconfigured per channel)
    adc_fifo_setup(
        true,   // enable FIFO
        true,   // enable DMA request (DREQ)
        1,      // DREQ when >=1 sample in FIFO
        false,  // don't set ERR bit
        false   // no 8-bit conversion
    );

    // Set up DMA for each channel (config only, don't start yet)
    void (*handlers[NUM_CHANNELS])(void) = {dma_handler_ch0, dma_handler_ch1, dma_handler_ch2};

    for (int i = 0; i < NUM_CHANNELS; i++) {
        dma_channel_config cfg = dma_channel_get_default_config(channels[i].dma_chan);
        channel_config_set_transfer_data_size(&cfg, DMA_SIZE_16);
        channel_config_set_read_increment(&cfg, false);
        channel_config_set_write_increment(&cfg, true);
        channel_config_set_dreq(&cfg, DREQ_ADC);

        dma_channel_configure(
            channels[i].dma_chan,
            &cfg,
            channels[i].adc_buffer[0],
            &adc_hw->fifo,
            FFT_SIZE,
            false
        );

        // Set up interrupt for this DMA channel
        dma_channel_set_irq0_enabled(channels[i].dma_chan, true);
    }

    // Set handlers for DMA IRQ (use shared handlers)
    irq_set_exclusive_handler(DMA_IRQ_0, dma_handler_ch0);
    irq_add_shared_handler(DMA_IRQ_0, dma_handler_ch1, PICO_SHARED_IRQ_HANDLER_DEFAULT_ORDER_PRIORITY);
    irq_add_shared_handler(DMA_IRQ_0, dma_handler_ch2, PICO_SHARED_IRQ_HANDLER_DEFAULT_ORDER_PRIORITY);
    irq_set_enabled(DMA_IRQ_0, true);
}

// Helper function to start sampling on a specific channel
void start_channel_sampling(int ch_idx) {
    channel_t *ch = &channels[ch_idx];

    // Stop ADC
    adc_run(false);
    adc_fifo_drain();

    // Select this ADC input
    adc_select_input(ch->adc_channel);

    // Set sample rate for this channel
    adc_set_clkdiv(48000000.0f / ch->sample_rate);

    // Start ADC
    adc_run(true);
}

// ------------------ Audio processing for one buffer from a specific channel ------------------
void process_audio_from_buffer(uint16_t *buf, int ch_idx) {
    channel_t *ch = &channels[ch_idx];

    // compute DC offset
    float dc = 0.0f;
    for (int i = 0; i < FFT_SIZE; i++) dc += buf[i];
    dc /= (float)FFT_SIZE;

    // convert to floats and normalize to approx ±1.0 (12-bit)
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_input[i] = ((float)buf[i] - dc) / 2048.0f;
    }

    // window and FFT
    apply_hanning_window(fft_input, FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_real[i] = fft_input[i];
        fft_imag[i] = 0.0f;
    }
    fft(fft_real, fft_imag, FFT_SIZE);
    calculate_magnitude(fft_real, fft_imag, magnitude, FFT_SIZE);

    // Map this channel's frequency range to the combined spectrum
    map_channel_to_bands(magnitude, combined_spectrum, FFT_SIZE, NUM_BANDS,
                        ch->sample_rate, ch->min_freq, ch->max_freq);
}

// ------------------ Main ------------------
int main() {
    stdio_init_all();
    sleep_ms(1000);

    // LED heartbeat
    gpio_init(23);
    gpio_set_dir(23, GPIO_OUT);
    gpio_put(23, 1);

    printf("\n=== 3-Channel FFT: 20Hz - 20kHz ===\n");
    printf("Ch0: 20-7447Hz @ 16kHz | Ch1: 7242-14773Hz @ 32kHz | Ch2: 14573-22kHz @ 48kHz\n");

    setup_adc_dma();

    // Initialize combined spectrum to zero
    for (int i = 0; i < NUM_BANDS; i++) {
        combined_spectrum[i] = 0.0f;
    }

    // Track which channel is currently sampling
    static int active_channel = 0;
    static bool channel_running[NUM_CHANNELS] = {false, false, false};

    // Start all channels with staggered timing (10ms offsets)
    // Channel 0 (lowest freq, highest priority) starts first
    for (int i = 0; i < NUM_CHANNELS; i++) {
        start_channel_sampling(i);
        dma_channel_start(channels[i].dma_chan);
        channel_running[i] = true;
        sleep_ms(10);  // Stagger by 10ms
    }

    printf("All channels started. Processing...\n");

    uint32_t vis_counter = 0;
    const uint32_t VIS_INTERVAL = 3;  // Visualize every 3 FFTs

    while (1) {
        // Priority processing: check channels in order 0, 1, 2 (low freq to high freq)
        for (int i = 0; i < NUM_CHANNELS; i++) {
            if (channels[i].buffer_ready) {
                // Take ownership of the filled buffer
                int bufid = channels[i].filled_buffer_index;
                channels[i].buffer_ready = false;
                channels[i].filled_buffer_index = -1;

                // Process FFT for this channel
                process_audio_from_buffer(channels[i].adc_buffer[bufid], i);

                // Reconfigure DMA to the other buffer for the next transfer
                int next = bufid ^ 1;

                // Restart this channel's ADC and DMA
                start_channel_sampling(i);
                dma_channel_set_write_addr(channels[i].dma_chan,
                                          channels[i].adc_buffer[next], true);

                // Visualize spectrum periodically (not every single FFT to reduce overhead)
                vis_counter++;
                if (vis_counter >= VIS_INTERVAL) {
                    visualize_spectrum(combined_spectrum, NUM_BANDS);
                    vis_counter = 0;

                    // Decay the spectrum slightly for next update
                    // This creates a "persistence" effect
                    for (int j = 0; j < NUM_BANDS; j++) {
                        combined_spectrum[j] *= 0.7f;
                    }
                }

                // Only process one channel per loop iteration to maintain priority
                break;
            }
        }

        tight_loop_contents();
    }

    return 0;
}
//test