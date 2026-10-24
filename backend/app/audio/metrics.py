import numpy as np
import scipy.io.wavfile as wavfile
import io
import math
import wave
from typing import Tuple, Dict, Any, Optional
from ..schemas.schemas import AudioQualityMetrics

# Try importing soundfile for comprehensive local format support (WAV, MP3, FLAC, OGG)
try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    HAVE_SOUNDFILE = False

def parse_audio_data(file_bytes: bytes, filename: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Parses audio bytes into a float32 mono numpy array normalized to [-1.0, 1.0],
    along with its sampling rate.
    Supports WAV, MP3, FLAC, and OGG via soundfile and scipy.
    Guarantees no raw compressed bytes are mistaken for PCM samples.
    """
    if not file_bytes or len(file_bytes) < 64:
        raise ValueError("Audio file is empty or too small to contain valid audio data.")

    if len(file_bytes) > 50 * 1024 * 1024:
        raise ValueError("Audio file exceeds maximum allowed size of 50 MB.")

    samples: Optional[np.ndarray] = None
    sr: Optional[int] = None
    errors = []

    # 1. Primary decoder: soundfile (handles WAV, MP3, FLAC, OGG)
    if HAVE_SOUNDFILE:
        try:
            data, sample_rate = sf.read(io.BytesIO(file_bytes), dtype='float32', always_2d=False)
            samples = data
            sr = int(sample_rate)
        except Exception as e:
            errors.append(f"soundfile: {e}")

    # 2. Fallback decoder: scipy.io.wavfile (standard uncompressed WAV)
    if samples is None:
        try:
            sample_rate, data = wavfile.read(io.BytesIO(file_bytes))
            sr = int(sample_rate)
            # Normalize depending on input dtype
            if data.dtype == np.int16:
                samples = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                samples = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                samples = (data.astype(np.float32) - 128.0) / 128.0
            else:
                samples = data.astype(np.float32)
        except Exception as e:
            errors.append(f"scipy.wavfile: {e}")

    # 3. Fallback decoder: standard library wave module (PCM WAV)
    if samples is None:
        try:
            with wave.open(io.BytesIO(file_bytes), 'rb') as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw_frames = wf.readframes(n_frames)

                if sampwidth == 2:
                    raw_arr = np.frombuffer(raw_frames, dtype=np.int16)
                    samples = raw_arr.astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    raw_arr = np.frombuffer(raw_frames, dtype=np.int32)
                    samples = raw_arr.astype(np.float32) / 2147483648.0
                elif sampwidth == 1:
                    raw_arr = np.frombuffer(raw_frames, dtype=np.uint8)
                    samples = (raw_arr.astype(np.float32) - 128.0) / 128.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

                if n_channels > 1:
                    samples = samples.reshape(-1, n_channels)
        except Exception as e:
            errors.append(f"wave: {e}")

    # If all decoders failed, raise a clear error; do NOT interpret compressed bits as raw PCM!
    if samples is None or sr is None:
        name_hint = f" for '{filename}'" if filename else ""
        raise ValueError(
            f"Unable to decode audio format{name_hint}. Please ensure the file is an intact audio recording "
            f"(WAV, MP3, FLAC, or OGG). Parsing details: {'; '.join(errors)}"
        )

    # Convert multi-channel to mono cleanly
    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)

    # Sanitize NaN / Inf
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # Validate minimal length
    if len(samples) < 160: # less than 10ms at 16kHz
        raise ValueError("Audio recording is too short to extract meaningful acoustic measurements.")

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

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")

    duration_sec = float(N) / float(sample_rate)

    # 1. Peak Amplitude [0.0, 1.0]
    peak_amp = float(np.max(np.abs(samples))) if N > 0 else 0.0

    # 2. RMS Loudness in dBFS
    # 0 dBFS = digital full scale square wave
    rms = float(np.sqrt(np.mean(samples ** 2)))
    rms_dbfs = 20.0 * math.log10(max(1e-9, rms))
    rms_dbfs = max(-96.0, min(0.0, rms_dbfs))

    # 3. Clipping Percentage (|sample| >= 0.98)
    clipped_count = np.count_nonzero(np.abs(samples) >= 0.98)
    clipping_pct = float((clipped_count / N) * 100.0)

    # 4. Frame-based analysis for VAD, SNR, and Background Noise
    # Frame size = 25ms, Hop size = 10ms
    frame_len = max(16, int(sample_rate * 0.025))
    hop_len = max(8, int(sample_rate * 0.010))

    if N < frame_len:
        frame_energies = np.array([rms ** 2], dtype=np.float32)
        zcr = np.array([0.0], dtype=np.float32)
    else:
        num_frames = 1 + (N - frame_len) // hop_len
        frame_energies = np.zeros(num_frames, dtype=np.float32)
        zcr = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * hop_len
            frame = samples[start:start + frame_len]
            frame_energies[i] = np.mean(frame ** 2)
            signs = np.sign(frame)
            zcr[i] = np.mean(np.abs(signs[1:] - signs[:-1])) / 2.0

    # 5. Background noise estimation: 10th percentile energy frame
    sorted_energies = np.sort(frame_energies)
    noise_energy = max(1e-9, float(np.percentile(sorted_energies, 10)))
    background_noise_db = 10.0 * math.log10(noise_energy)
    background_noise_db = max(-96.0, min(0.0, background_noise_db))

    # 6. Speech activity estimation: Energy threshold above noise floor
    speech_threshold = max(noise_energy * 4.0, 1e-5)
    speech_mask = (frame_energies > speech_threshold) & (zcr < 0.45)
    speech_frames = np.count_nonzero(speech_mask)
    total_frames = max(1, len(frame_energies))
    speech_ratio = float(speech_frames / total_frames)
    silence_ratio = float(1.0 - speech_ratio)

    # 7. Estimated Signal to Noise Ratio (SNR) in dB
    # Ratio of top 20% active speech energy to quietest 10% noise floor
    speech_top_energy = max(1e-9, float(np.percentile(sorted_energies, 80)))
    if speech_top_energy > noise_energy:
        snr_db = 10.0 * math.log10(speech_top_energy / noise_energy)
    else:
        snr_db = 0.0
    snr_db = max(0.0, min(60.0, snr_db))

    # 8. Spectral Centroid & Spectral Rolloff using FFT
    fft_vals = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(N, 1.0 / sample_rate)

    fft_sum = np.sum(fft_vals)
    if fft_sum > 1e-9:
        spectral_centroid = float(np.sum(freqs * fft_vals) / fft_sum)
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
