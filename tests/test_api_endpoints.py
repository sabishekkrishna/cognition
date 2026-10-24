import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import io
import scipy.io.wavfile as wavfile
from backend.app.main import app

client = TestClient(app)

def test_api_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "active_vision_model" in data

def test_api_status_endpoint():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["overall_state"] in ["normal", "caution", "warning", "unavailable"]
    assert "overall_reliability_score" in data

def test_api_image_sample_and_analyze():
    # 1. Get sample image
    sample_res = client.get("/api/image/sample")
    assert sample_res.status_code == 200
    img_bytes = sample_res.content

    # 2. Analyze image
    analyze_res = client.post(
        "/api/image/analyze",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    )
    assert analyze_res.status_code == 200
    data = analyze_res.json()
    assert "metrics" in data
    assert "predictions" in data
    assert "reliability_score" in data
    assert 0 <= data["reliability_score"] <= 100

def test_api_image_degrade():
    sample_res = client.get("/api/image/sample")
    img_bytes = sample_res.content

    degrade_res = client.post(
        "/api/image/degrade",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        data={"gaussian_blur_kernel": 21, "brightness_factor": 0.3}
    )
    assert degrade_res.status_code == 200
    data = degrade_res.json()
    assert data["is_synthetically_degraded"] is True
    assert "degraded_analysis" in data
    assert "original_analysis" in data
    # Degraded score should be lower than original
    assert data["degraded_analysis"]["reliability_score"] <= data["original_analysis"]["reliability_score"]

def test_api_audio_sample_and_analyze():
    sample_res = client.get("/api/audio/sample")
    assert sample_res.status_code == 200
    audio_bytes = sample_res.content

    analyze_res = client.post(
        "/api/audio/analyze",
        files={"file": ("sample.wav", io.BytesIO(audio_bytes), "audio/wav")}
    )
    assert analyze_res.status_code == 200
    data = analyze_res.json()
    assert "metrics" in data
    assert "classification" in data
    assert 0 <= data["reliability_score"] <= 100

def test_api_llm_providers_and_generate():
    prov_res = client.get("/api/llm/providers")
    assert prov_res.status_code == 200
    providers = prov_res.json()["providers"]
    assert any(p["id"] == "ollama" for p in providers)
    assert any(p["id"] == "mock" for p in providers)

    # Generate with local mock provider
    gen_res = client.post(
        "/api/llm/generate",
        json={
            "provider": "mock",
            "model": "demo-llama3-8b-instruct (Synthetic)",
            "prompt": "Test inspection prompt",
            "current_input_risk_score": 10.0
        }
    )
    assert gen_res.status_code == 200
    data = gen_res.json()
    assert len(data["response"]) > 0
    assert "interaction_id" in data

    # Test feedback on that interaction
    fb_res = client.post(
        "/api/llm/feedback",
        json={
            "interaction_id": data["interaction_id"],
            "user_rating": "correct",
            "user_notes": "Good test response"
        }
    )
    assert fb_res.status_code == 200

def test_api_baselines_list():
    res = client.get("/api/baselines")
    assert res.status_code == 200
    baselines = res.json()
    assert len(baselines) >= 2 # image & audio default baselines
