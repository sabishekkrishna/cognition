import numpy as np
import scipy.io.wavfile as wavfile
import io
from typing import Tuple
from ..schemas.schemas import AudioDegradationConfig

def apply_audio_degradation(samples: np.ndarray, sample_rate: int,
                           config: AudioDegradationConfig) -> Tuple[np.ndarray, str]:
    """
    Applies synthetic acoustic degradations to audio samples:
    - Volume scaling (attenuation or excessive gain)
    - Additive noise (white noise, pink noise, low rumble)
    - Hard clipping
    Returns: (degraded_samples_float32, description)
    """
    degraded = samples.copy().astype(np.float32)
    effects = []

    # 1. Volume factor
    if config.volume_factor != 1.0:
        degraded = degraded * config.volume_factor
        if config.volume_factor < 1.0:
            effects.append(f"Volume Attenuation ({int(config.volume_factor * 100)}%)")
        else:
            effects.append(f"Volume Boost ({int(config.volume_factor * 100)}%)")

    # 2. Additive Noise
    if config.noise_level > 0.0:
        lvl = float(np.clip(config.noise_level, 0.0, 1.0))
        N = len(degraded)

        if config.noise_type == "low_rumble":
            # Low frequency sinusoidal/rumble noise (e.g. 80Hz + harmonics)
            t = np.linspace(0, N / sample_rate, N, endpoint=False)
            noise = (np.sin(2 * np.pi * 65 * t) + 0.5 * np.sin(2 * np.pi * 130 * t)).astype(np.float32)
            noise += 0.2 * np.random.normal(0, 1.0, N).astype(np.float32)
            effects.append(f"Low-Frequency Wind/Rumble Noise ({int(lvl * 100)}%)")
        elif config.noise_type == "pink":
            # Simple 1/f approximation
            white = np.random.normal(0, 1.0, N).astype(np.float32)
            # Cumulative moving average creates low-frequency pink-ish bias
            pink = np.cumsum(white) / np.sqrt(N)
            pink = pink - np.mean(pink)
            std = np.std(pink)
            noise = (pink / (std + 1e-6)).astype(np.float32)
            effects.append(f"Pink Acoustic Noise ({int(lvl * 100)}%)")
        else:
            # White Gaussian noise
            noise = np.random.normal(0, 1.0, N).astype(np.float32)
            effects.append(f"White Static Noise ({int(lvl * 100)}%)")

        degraded = degraded + noise * (lvl * 0.35)

    # 3. Non-linear clipping / saturation
    if config.clipping_gain > 1.0:
        degraded = degraded * config.clipping_gain
        degraded = np.clip(degraded, -1.0, 1.0)
        effects.append(f"Waveform Saturation & Clipping ({config.clipping_gain:.1f}x gain)")
    else:
        degraded = np.clip(degraded, -1.0, 1.0)

    summary = ", ".join(effects) if effects else "None (Clean Original)"
    return degraded, summary

def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encodes float32 samples into standard 16-bit PCM WAV bytes."""
    samples_clipped = np.clip(samples, -1.0, 1.0)
    int16_samples = (samples_clipped * 32767.0).astype(np.int16)
    out_io = io.BytesIO()
    wavfile.write(out_io, sample_rate, int16_samples)
    return out_io.getvalue()
