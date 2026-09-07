"""Noise mixing helpers for ASR robustness tests."""

from __future__ import annotations

import numpy as np


def add_white_noise_snr(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix white noise at a target Signal-to-Noise Ratio (SNR) in decibels (dB).

    SNR here means: speech power / noise power.
    Higher dB = cleaner audio (example: 20 dB is mild noise, 5 dB is very noisy).
    """
    if snr_db <= 0:
        raise ValueError("snr_db must be positive.")

    signal = signal.astype(np.float32)
    if signal.size == 0:
        return signal

    signal_power = np.mean(signal**2)
    if signal_power == 0:
        return signal

    noise = np.random.randn(signal.size).astype(np.float32)
    noise_power = np.mean(noise**2)

    # target noise power from SNR formula: SNR(dB) = 10 * log10(signal_power / noise_power)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / max(noise_power, 1e-12))
    mixed = signal + scaled_noise

    return np.clip(mixed, -1.0, 1.0).astype(np.float32)
