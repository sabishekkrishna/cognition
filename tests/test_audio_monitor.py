import pytest
import numpy as np
from backend.app.audio.metrics import analyze_audio_quality, parse_audio_data
from backend.app.audio.classifier import get_audio_classifier
from backend.app.audio.degradation import apply_audio_degradation, samples_to_wav_bytes
from backend.app.schemas.schemas import AudioDegradationConfig

def create_synthetic_audio(duration: float = 1.0, sr: int = 16000, freq: float = 300.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Speech harmonic tone with silence at end for accurate SNR contrast
    audio = np.zeros_like(t, dtype=np.float32)
    active_len = int(len(t) * 0.7)
    audio[:active_len] = (0.5 * np.sin(2 * np.pi * freq * t[:active_len]) + 
                          0.25 * np.sin(2 * np.pi * freq * 2 * t[:active_len])).astype(np.float32)
    return audio

def test_audio_metrics_calculation():
    audio = create_synthetic_audio(1.0, 16000)
    metrics = analyze_audio_quality(audio, 16000)

    assert -80.0 <= metrics.rms_dbfs <= 0.0
    assert 0.0 <= metrics.peak_amplitude <= 1.0
    assert metrics.clipping_pct == 0.0
    assert metrics.duration_sec == pytest.approx(1.0, rel=0.05)
    assert metrics.sample_rate == 16000
    assert metrics.spectral_centroid_hz > 0

def test_audio_clipping_detection():
    audio = create_synthetic_audio(1.0, 16000)
    config = AudioDegradationConfig(clipping_gain=5.0)
    clipped_audio, desc = apply_audio_degradation(audio, 16000, config)
    metrics = analyze_audio_quality(clipped_audio, 16000)

    assert metrics.clipping_pct > 5.0
    assert "Clipping" in desc

def test_audio_noise_detection():
    audio = create_synthetic_audio(1.0, 16000)
    clean_metrics = analyze_audio_quality(audio, 16000)

    config = AudioDegradationConfig(noise_type="white", noise_level=0.8)
    noisy_audio, _ = apply_audio_degradation(audio, 16000, config)
    noisy_metrics = analyze_audio_quality(noisy_audio, 16000)

    assert noisy_metrics.snr_db < clean_metrics.snr_db

def test_acoustic_condition_classifier():
    classifier = get_audio_classifier()
    audio = create_synthetic_audio(1.0, 16000)
    metrics = analyze_audio_quality(audio, 16000)

    res = classifier.classify(
        audio, 16000,
        peak_amp=metrics.peak_amplitude,
        clipping_pct=metrics.clipping_pct,
        rms_dbfs=metrics.rms_dbfs,
        snr_db=metrics.snr_db,
        spectral_centroid=metrics.spectral_centroid_hz
    )

    assert res.primary_condition in res.condition_probabilities
    assert sum(res.condition_probabilities.values()) == pytest.approx(1.0, abs=0.01)
