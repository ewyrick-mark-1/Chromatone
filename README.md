# Chromatone

**Author:** Elliot Wyrick

## Overview

Chromatone is an embedded audio visualizer running on the Raspberry Pi RP2350. It samples audio through the ADC, performs a real-time FFT, detects musical notes, and drives an RGB LED output based on the results.

## [Demo video:](https://youtu.be/pOVmz1aZbJg)

[![Watch the demo](https://img.youtube.com/vi/pOVmz1aZbJg/maxresdefault.jpg)](https://youtu.be/pOVmz1aZbJg)


## Hardware

- **MCU:** Raspberry Pi RP2350 (overclocked to 300 MHz)
- **Input:** Analog microphone on ADC channel 0 (GPIO 40)
- **Output:** RGB LED via PWM on GPIO 37–39
- **Schematics/PCB:** KiCad project files located in `KiCad/`

## How It Works

Audio is captured at 20 kHz using DMA ping-pong buffers. Each buffer triggers a 16384 point FFT with 50% overlap and a Hanning window applied. Detected frequency peaks are mapped to MIDI notes (A0–C8), and the corresponding hue is output to the RGB LED.

Two display modes are available (toggled via `display_mode` in `main.c`):
- **Mode 0:** Logarithmic frequency spectrum
- **Mode 1:** Musical note detection

## Color Math

Each of the 12 chromatic notes is assigned a fixed hue evenly spaced around the color wheel (360° / 12 = 30° per note), with C at 0° (red) through B at 330° (pink).

Detected peaks are converted to a note index using the standard MIDI formula:

```
n = 12 × log₂(f / 440) + 69
```

When multiple notes are detected simultaneously, their hues are blended using a **circular weighted average**- each hue is converted to a unit vector on the color wheel, weighted by FFT magnitude, and the vectors are summed:

```
x = Σ( cos(hue_i) × weight_i )
y = Σ( sin(hue_i) × weight_i )
blended_hue = atan2(y, x)
```

This handles wraparound correctly (e.g. blending 10° and 350° yields 0°, not 180°).

The blended hue is then smoothed over time using an exponential moving average on the shortest angular path (15% per frame) to avoid jarring color jumps between notes.

Finally, the smoothed hue is converted from HSV (full saturation and brightness) to RGB via the standard piecewise HSV→RGB formula and output as an 8-bit PWM duty cycle on each channel.

## Build & Flash

This project uses [PlatformIO](https://platformio.org/) with the Proton board target.

```
pio run
pio run --target upload
```

Serial monitor at 115200 baud:
```
pio device monitor
```

Alternativly, and what I used during development, the platform IO extension works well. It provides gui upload and monitor buttons.
