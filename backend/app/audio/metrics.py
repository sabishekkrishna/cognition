import numpy as np
import scipy.io.wavfile as wavfile
import io
import math
from typing import Tuple, Dict, Any
from ..schemas.schemas import AudioQualityMetrics

def parse_audio_data(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Parses WAV audio bytes into a float32 mono numpy array normalized to [-1.0, 1.0],
    along with its sampling rate.
    """
    try:
        sr, data = wavfile.read(io.BytesIO(file_bytes))
    except Exception as e:
        # If header fails, attempt reading 16-bit PCM raw buffer at 16000Hz default
        raw = np.frombuffer(file_bytes, dtype=np.int16)
        if len(raw) > 0:
            sr = 16000
            data = raw
        else:
            raise ValueError(f"Unable to parse audio stream: {e}")

    # Convert multi-channel to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalize to float32 [-1.0, 1.0] depending on input dtype
    if data.dtype == np.int16:
        samples = (data.astype(np.float32)) / 32768.0
    elif data.dtype == np.int32:
        samples = (data.astype(np.float32)) / 2147483648.0
    elif data.dtype == np.uint8:
        samples = (data.astype(np.float32) - 128.0) / 128.0
    else:
        samples = data.astype(np.float32)

    return samples, int(sr)

def analyze_audio_quality(samples: np.ndarray, sample_rate: int) -> AudioQualityMetrics:
    """
    Computes local acoustic signal quality measurements:
    - RMS loudness (dBFS)
    - Peak amplitude
    - Clipping percentage
    - Background noise level
    - Estimated Signal-to-Noise Ratio (SNR)
    - Voice Activity Detection (VAD) speech & silence proportions
    - Spectral Centroid & Spectral Rolloff
    """
    N = len(samples)
    if N == 0:
        raise ValueError("Audio sample array is empty.")

    duration_sec = float(N) / float(sample_rate)

    # 1. Peak Amplitude [0.0, 1.0]
    peak_amp = float(np.max(np.abs(samples)))

    # 2. RMS Loudness in dBFS
    # 0 dBFS = digital full scale square wave, sine full scale = -3.01 dBFS
    rms = float(np.sqrt(np.mean(samples ** 2)))
    rms_dbfs = 20.0 * math.log10(rms + 1e-9)

    # 3. Clipping Percentage (|sample| >= 0.98)
    clipped_count = np.count_nonzero(np.abs(samples) >= 0.98)
    clipping_pct = float((clipped_count / N) * 100.0)

    # 4. Frame-based analysis for VAD, SNR, and Background Noise
    # Frame size = 25ms, Hop size = 10ms
    frame_len = int(sample_rate * 0.025)
    hop_len = int(sample_rate * 0.010)

    if N < frame_len:
        # Very short clip fallback
        frame_energies = np.array([rms ** 2])
        zcr = np.array([0.0])
    else:
        num_frames = 1 + (N - frame_len) // hop_len
        frame_energies = np.zeros(num_frames, dtype=np.float32)
        zcr = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * hop_len
            frame = samples[start:start + frame_len]
            frame_energies[i] = np.mean(frame ** 2)
            # Zero crossing rate
            signs = np.sign(frame)
            zcr[i] = np.mean(np.abs(signs[1:] - signs[:-1])) / 2.0

    # 5. Background noise estimation: 10th percentile energy frame
    sorted_energies = np.sort(frame_energies)
    noise_energy = float(np.percentile(sorted_energies, 10)) + 1e-9
    background_noise_db = 10.0 * math.log10(noise_energy)

    # 6. Speech activity estimation: Energy threshold above noise floor
    # Speech frames typically > 10dB above noise floor or threshold
    speech_threshold = max(noise_energy * 5.0, 1e-5)
    speech_mask = (frame_energies > speech_threshold) & (zcr < 0.45)
    speech_frames = np.count_nonzero(speech_mask)
    total_frames = len(frame_energies)
    speech_ratio = float(speech_frames / total_frames)
    silence_ratio = float(1.0 - speech_ratio)

    # 7. Estimated Signal to Noise Ratio (SNR) in dB
    # Ratio of top 20% active speech energy to quietest 10% noise floor
    speech_top_energy = float(np.percentile(sorted_energies, 80)) + 1e-9
    snr_db = 10.0 * math.log10(speech_top_energy / noise_energy)
    snr_db = max(0.0, min(60.0, snr_db)) # Reasonable physical acoustic bounds

    # 8. Spectral Centroid & Spectral Rolloff using FFT
    fft_vals = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(N, 1.0 / sample_rate)

    fft_sum = np.sum(fft_vals)
    if fft_sum > 1e-9:
        spectral_centroid = float(np.sum(freqs * fft_vals) / fft_sum)

        # Spectral rolloff: frequency below which 85% of total magnitude energy is contained
        cumulative_energy = np.cumsum(fft_vals)
        rolloff_idx = np.searchsorted(cumulative_energy, 0.85 * fft_sum)
        spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    else:
        spectral_centroid = 0.0
        spectral_rolloff = 0.0

    return AudioQualityMetrics(
        rms_dbfs=round(rms_dbfs, 2),
        peak_amplitude=round(peak_amp, 3),
        clipping_pct=round(clipping_pct, 2),
        background_noise_db=round(background_noise_db, 2),
        snr_db=round(snr_db, 2),
        speech_ratio=round(speech_ratio, 3),
        silence_ratio=round(silence_ratio, 3),
        spectral_centroid_hz=round(spectral_centroid, 1),
        spectral_rolloff_hz=round(spectral_rolloff, 1),
        duration_sec=round(duration_sec, 2),
        sample_rate=int(sample_rate)
    )
