import pytest
import io
import wave
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.audio.metrics import parse_audio_data, analyze_audio_quality
from backend.app.monitoring.engine import get_monitoring_engine, UnifiedMonitoringEngine
from backend.app.baselines.manager import BaselineManager, compute_robust_distribution, DOMAIN_MIN_MAD
from backend.app.llm_providers.base import parse_and_validate_local_endpoint
from backend.app.llm_providers.registry import get_provider_registry
from backend.app.schemas.schemas import ImageQualityMetrics, ModelPredictions, PredictionItem, AudioQualityMetrics

client = TestClient(app)

def create_synthetic_wav_bytes(duration: float = 1.0, sr: int = 16000, freq: float = 440.0, num_channels: int = 1) -> bytes:
    """Helper to synthesize valid PCM WAV bytes."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if num_channels == 2:
        sig = np.column_stack([sig, sig])
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()

def create_synthetic_mp3_bytes(duration: float = 1.0, sr: int = 16000, freq: float = 440.0) -> bytes:
    """Helper to synthesize valid OGG/FLAC audio bytes for multi-format testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format='OGG')
    return buf.getvalue()

# ==================== ISSUE 1 TESTS: DASHBOARD UPDATES ====================

def test_issue1_dashboard_updates_on_discrete_image_analysis():
    """Verify that discrete image analysis immediately updates metrics and does not hide warnings behind stream hysteresis."""
    engine = UnifiedMonitoringEngine()
    
    # Degraded dark and blurry image metrics
    degraded_metrics = ImageQualityMetrics(
        brightness=15.0, # severely underexposed
        sharpness=10.0,  # severely blurry
        contrast=8.0,
        color_mean_r=15.0,
        color_mean_g=15.0,
        color_mean_b=15.0,
        color_cast_delta=0.0,
        noise_estimate=2.0,
        underexposure_pct=85.0,
        overexposure_pct=0.0,
        width=640,
        height=480,
        resolution_mp=0.31
    )
    predictions = ModelPredictions(
        model_name="mobilenetv3_small_v1",
        top_predictions=[PredictionItem(class_id=0, class_name="test", probability=0.3)],
        top_class="test",
        top_probability=0.3,
        entropy=4.2,
        margin=0.05,
        embedding_dimension=1024
    )
    
    # Process with is_stream=False (discrete upload)
    res = engine.process_image_frame(degraded_metrics, predictions, is_stream=False)
    assert res["status"] in ["warning", "caution"], "Discrete degraded image must immediately register as warning/caution"
    assert engine.image_state in ["warning", "caution"], "Engine image state must be updated without requiring multiple frames"
    assert engine.latest_image_metrics == degraded_metrics
    assert len(engine.active_image_warnings) > 0, "Active image warnings must be populated"

    overall = engine.get_system_overall_status()
    assert overall.overall_state in ["warning", "caution"], "Overall system status must reflect the degraded subsystem"
    assert overall.last_image_metrics is not None
    assert overall.last_updated_formatted is not None

def test_issue1_subsystems_warnings_do_not_overwrite_each_other():
    """Verify image warnings and audio warnings do not erase each other."""
    engine = UnifiedMonitoringEngine()
    
    img_metrics = ImageQualityMetrics(
        brightness=10.0, sharpness=10.0, contrast=5.0,
        color_mean_r=10.0, color_mean_g=10.0, color_mean_b=10.0,
        color_cast_delta=0.0, noise_estimate=1.0,
        underexposure_pct=90.0, overexposure_pct=0.0,
        width=100, height=100, resolution_mp=0.01
    )
    preds = ModelPredictions(
        model_name="m", top_predictions=[], top_class="c",
        top_probability=0.2, entropy=5.0, margin=0.01, embedding_dimension=1024
    )
    engine.process_image_frame(img_metrics, preds, is_stream=False)
    assert len(engine.active_image_warnings) > 0

    aud_metrics = AudioQualityMetrics(
        rms_dbfs=-6.0, peak_amplitude=1.0, clipping_pct=25.0,
        background_noise_db=-20.0, snr_db=5.0, speech_ratio=0.5,
        silence_ratio=0.5, spectral_centroid_hz=2000.0, spectral_rolloff_hz=3500.0,
        duration_sec=2.0, sample_rate=16000
    )
    from backend.app.schemas.schemas import AudioClassificationResult
    aud_class = AudioClassificationResult(
        model_name="cnn", primary_condition="clipping", condition_probabilities={"clipping": 0.9}
    )
    engine.process_audio_chunk(aud_metrics, aud_class, is_stream=False)
    assert len(engine.active_audio_warnings) > 0
    assert len(engine.active_image_warnings) > 0, "Audio processing must not clear active image warnings"
    assert len(engine.active_warnings) == len(engine.active_image_warnings) + len(engine.active_audio_warnings)

def test_issue1_monitoring_state_toggle():
    """Verify monitoring state can be paused/resumed and reflects unavailable state."""
    engine = UnifiedMonitoringEngine()
    engine.set_monitoring_active(False)
    status = engine.get_system_overall_status()
    assert status.monitoring_active is False
    assert status.overall_state == "unavailable"

    engine.set_monitoring_active(True)
    status2 = engine.get_system_overall_status()
    assert status2.monitoring_active is True

# ==================== ISSUE 2 TESTS: AUDIO UPLOAD & DECODING ====================

def test_issue2_wav_mono_and_stereo_decoding():
    """Verify WAV audio parsing handles both mono and stereo without distortion."""
    mono_bytes = create_synthetic_wav_bytes(duration=1.0, sr=16000, num_channels=1)
    samples_m, sr_m = parse_audio_data(mono_bytes, filename="mono.wav")
    assert sr_m == 16000
    assert len(samples_m) == 16000
    assert samples_m.ndim == 1

    stereo_bytes = create_synthetic_wav_bytes(duration=1.0, sr=16000, num_channels=2)
    samples_s, sr_s = parse_audio_data(stereo_bytes, filename="stereo.wav")
    assert sr_s == 16000
    assert len(samples_s) == 16000
    assert samples_s.ndim == 1, "Stereo audio must be cleanly converted to mono"

def test_issue2_compressed_ogg_flac_decoding():
    """Verify compressed audio formats decode cleanly via soundfile."""
    ogg_bytes = create_synthetic_mp3_bytes(duration=1.0, sr=16000)
    samples, sr = parse_audio_data(ogg_bytes, filename="sample.ogg")
    assert sr == 16000
    assert len(samples) > 0
    metrics = analyze_audio_quality(samples, sr)
    assert metrics.duration_sec >= 0.9
    assert -96.0 <= metrics.rms_dbfs <= 0.0

def test_issue2_corrupted_file_rejection_no_raw_byte_interpretation():
    """Verify corrupted or invalid binary files are rejected and NOT converted to raw PCM noise."""
    corrupted_data = b"RIFF" + b"\x00" * 20 + b"WAVEfmt " + b"\xff" * 50
    with pytest.raises(ValueError) as exc:
        parse_audio_data(corrupted_data, filename="corrupt.wav")
    assert "Unable to decode audio format" in str(exc.value)

def test_issue2_empty_file_rejection():
    """Verify 0-byte or tiny files are cleanly rejected with a clear message."""
    with pytest.raises(ValueError) as exc:
        parse_audio_data(b"", filename="empty.wav")
    assert "empty or too small" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        parse_audio_data(b"too small", filename="tiny.wav")
    assert "too small" in str(exc2.value)

# ==================== ISSUE 3 TESTS: AUDIO BASELINE & ZERO-MAD ====================

def test_issue3_zero_mad_fallback_to_domain_threshold():
    """Verify that identical samples (zero variance) do not produce 0.0 MAD or division-by-zero."""
    # Identical spectral centroids
    identical_values = [1800.0, 1800.0, 1800.0, 1800.0, 1800.0]
    stats = compute_robust_distribution(identical_values, metric_name="spectral_centroid_hz")
    assert stats["median"] == 1800.0
    assert stats["mad"] > 0.0, "MAD must never be 0.0"
    assert stats["mad"] == DOMAIN_MIN_MAD["spectral_centroid_hz"], "Zero MAD must fall back to domain minimum"
    assert "p10" in stats and "p90" in stats, "Robust distribution must compute 10th and 90th percentiles"

def test_issue3_audio_baseline_calibration_and_deviation():
    """Verify audio baseline calibration persists and computes finite deviations."""
    wav1 = create_synthetic_wav_bytes(duration=1.0, sr=16000, freq=300.0)
    wav2 = create_synthetic_wav_bytes(duration=1.0, sr=16000, freq=350.0)
    
    samples1, sr1 = parse_audio_data(wav1)
    samples2, sr2 = parse_audio_data(wav2)
    m1 = analyze_audio_quality(samples1, sr1)
    m2 = analyze_audio_quality(samples2, sr2)

    baseline = BaselineManager.calibrate_audio_baseline(
        name="Test Room Audio Baseline",
        description="Clean baseline test profile",
        metrics_list=[m1, m2],
        set_active=True
    )
    assert baseline.id.startswith("aud_base_")
    assert baseline.sample_count == 2
    assert "rms_dbfs" in baseline.stats

    # Test deviation computation against the calibrated baseline
    deviations = BaselineManager.compute_audio_deviations(m1, baseline.stats)
    for k, dev in deviations.items():
        assert not np.isnan(dev), f"Deviation for {k} must not be NaN"
        assert not np.isinf(dev), f"Deviation for {k} must not be Infinite"

# ==================== ISSUE 4 TESTS: LOCAL LLM CONNECTION & DIAGNOSTICS ====================

def test_issue4_ssrf_protection_loopback_restriction():
    """Verify SSRF protection blocks non-loopback IPs unless allow_remote is True."""
    # Loopback addresses (MUST PASS)
    for valid_addr in ["http://127.0.0.1:11434", "localhost:8080", "127.0.0.1:1234", "http://localhost:11434"]:
        ok, canon_url, host, err, port = parse_and_validate_local_endpoint(valid_addr, allow_remote=False)
        assert ok is True, f"Loopback address '{valid_addr}' should be accepted"

    # Non-loopback addresses (MUST FAIL with SSRF explanation)
    for blocked_addr in ["http://192.168.1.50:11434", "10.0.0.1:8080", "http://8.8.8.8:11434", "example.com:11434"]:
        ok, _, _, err, _ = parse_and_validate_local_endpoint(blocked_addr, allow_remote=False)
        assert ok is False, f"Non-loopback address '{blocked_addr}' must be blocked"
        assert "SSRF" in err or "Security" in err

    # Permitted when allow_remote is True
    ok, canon_url, host, _, _ = parse_and_validate_local_endpoint("http://192.168.1.50:11434", allow_remote=True)
    assert ok is True, "Remote address must be allowed when allow_remote=True"

def test_issue4_port_validation():
    """Verify ports are validated within 1-65535."""
    ok, _, _, err, _ = parse_and_validate_local_endpoint("http://127.0.0.1:70000")
    assert ok is False
    assert "65535" in err

    ok2, canon_url, _, _, port = parse_and_validate_local_endpoint("http://127.0.0.1:9090")
    assert ok2 is True
    assert port == 9090

def test_issue4_dynamic_provider_url_update():
    """Verify ProviderRegistry dynamically updates base URL without needing server restart."""
    registry = get_provider_registry()
    registry.update_provider_url("ollama", "http://127.0.0.1:22444")
    
    prov = registry.get_provider("ollama")
    assert getattr(prov, "base_url") == "http://127.0.0.1:22444"
    
    # list_available_providers must reflect dynamic URL
    providers = registry.list_available_providers()
    ollama_info = next(p for p in providers if p["id"] == "ollama")
    assert "22444" in ollama_info["name"]

    # Reset back to default
    registry.update_provider_url("ollama", "http://127.0.0.1:11434")

def test_issue4_test_connection_diagnostic_endpoint_mock():
    """Verify POST /api/llm/test-connection executes all 5 diagnostic steps."""
    resp = client.post("/api/llm/test-connection", json={
        "provider": "mock",
        "save_if_successful": True
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall_status"] == "connected"
    assert len(data["steps"]) == 5
    for step in data["steps"]:
        assert step["status"] == "pass"

def test_issue4_test_connection_diagnostic_endpoint_unreachable_port():
    """Verify POST /api/llm/test-connection reports clean failure and remedy when port is unreachable."""
    resp = client.post("/api/llm/test-connection", json={
        "provider": "ollama",
        "host": "127.0.0.1",
        "port": 59999, # Unused test port
        "save_if_successful": False
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall_status"] == "failed"
    step2 = next(s for s in data["steps"] if s["step_number"] == 2)
    assert step2["status"] == "fail"
    assert "Connection refused" in step2["message"] or "timed out" in step2["message"]
    assert step2["remedy"] is not None
