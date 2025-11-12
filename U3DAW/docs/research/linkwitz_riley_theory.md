# Linkwitz-Riley 4th Order Crossover Theory

## Overview
LR-4 filters = Butterworth squared
Provides perfect phase coherence at crossover

## Transfer Functions (Frequency Domain)
- Sub (LP @ 200Hz): H(f) = 1 / sqrt(1 + (f/fc)^8)
- Bass (BP 80-400Hz): Bandpass combination
- Mid (BP 400-2000Hz): Bandpass combination  
- High (HP @ 2000Hz): H(f) = (f/fc)^4 / sqrt(1 + (f/fc)^8)

## Implementation
Use cuFFT for frequency-domain filtering
4096-point FFT at 192kHz sample rate
