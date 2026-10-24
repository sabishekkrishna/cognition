from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse
import cv2
import numpy as np
import base64
import json
import io
import time
from typing import Optional, List, Dict, Any

from ..schemas.schemas import (
    ImageQualityMetrics, ModelPredictions, ImageAnalysisResponse,
    AudioQualityMetrics, AudioClassificationResult, AudioAnalysisResponse,
    ImageDegradationConfig, AudioDegradationConfig,
    BaselineCreateRequest, BaselineInfo,
    LLMGenerateRequest, LLMResponseInfo, LLMFeedbackRequest,
    SystemOverallStatus
)
from ..image.metrics import analyze_image_quality
from ..image.classifier import get_classifier
from ..image.degradation import apply_image_degradation
from ..audio.metrics import analyze_audio_quality, parse_audio_data
from ..audio.classifier import get_audio_classifier
from ..audio.degradation import apply_audio_degradation, samples_to_wav_bytes
from ..baselines.manager import BaselineManager
from ..monitoring.engine import get_monitoring_engine
from ..monitoring.scoring import ReliabilityScoringEngine
from ..llm_providers.registry import get_provider_registry
from ..llm_providers.response_monitor import ResponseMonitor
from ..datasets.manager import DatasetManager
from ..training.trainer import ModelTrainingManager
from ..storage.database import (
    get_all_settings, set_setting, get_recent_events, clear_events,
    get_baselines, get_active_baseline, set_active_baseline, delete_baseline
)
from ..llm_providers.base import parse_and_validate_local_endpoint, join_url_path
from ..llm_providers.custom_http import CustomLocalProvider
from pydantic import BaseModel
import httpx

class DiagnosticStepResult(BaseModel):
    step_number: int
    name: str
    status: str # 'pass', 'fail', 'warn'
    latency_ms: float
    message: str
    remedy: Optional[str] = None

class LLMDiagnosticReport(BaseModel):
    provider: str
    target_url: str
    host: str
    port: int
    overall_status: str # 'connected', 'warning', 'failed'
    steps: List[DiagnosticStepResult]
    discovered_models: List[str]
    sample_response: Optional[str] = None
    saved_as_active: bool = False

class LLMConnectionTestRequest(BaseModel):
    provider: str = "ollama" # ollama, lm_studio, llama_cpp, custom_http, mock
    base_url: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    api_path: Optional[str] = None
    adapter_type: Optional[str] = "auto"
    model: Optional[str] = None
    allow_remote: bool = False
    save_if_successful: bool = False

api_router = APIRouter()

# Helper to decode image
def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")
    return img

def encode_image_base64(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode("utf-8")

# ==================== SYSTEM & HEALTH ====================

@api_router.get("/health")
async def health_check():
    engine = get_monitoring_engine()
    classifier = get_classifier()
    provider_reg = get_provider_registry()
    active_prov = provider_reg.get_provider()
    prov_ok, prov_msg = await active_prov.check_health()

    return {
        "status": "healthy",
        "service": "On-Device AI Reliability Monitor",
        "timestamp": time.time(),
        "active_vision_model": classifier.active_version,
        "active_llm_provider": active_prov.get_provider_name(),
        "llm_provider_connected": prov_ok,
        "llm_provider_message": prov_msg,
        "overall_state": engine.get_system_overall_status().overall_state
    }

@api_router.get("/status", response_model=SystemOverallStatus)
async def get_system_status():
    engine = get_monitoring_engine()
    return engine.get_system_overall_status()

# ==================== IMAGE SUBSYSTEM ====================

@api_router.post("/image/analyze", response_model=ImageAnalysisResponse)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    img_bgr = decode_image_bytes(content)

    # 1. OpenCV metrics
    metrics = analyze_image_quality(img_bgr)

    # 2. MobileNetV3 predictions
    classifier = get_classifier()
    predictions = classifier.predict(img_bgr, top_k=5)

    # 3. Baseline comparison & scoring via Unified Engine
    engine = get_monitoring_engine()
    res = engine.process_image_frame(metrics, predictions, is_stream=False)

    baseline = BaselineManager.get_or_create_default_image_baseline()

    return ImageAnalysisResponse(
        metrics=metrics,
        predictions=predictions,
        baseline_id=baseline["id"] if baseline else None,
        baseline_deviations=res["deviations"],
        reliability_score=res["reliability_score"],
        risk_score=res["risk_score"],
        status=res["status"],
        warnings=res["warnings"],
        is_synthetically_degraded=False
    )

@api_router.post("/image/degrade")
async def degrade_and_analyze_image(
    file: UploadFile = File(...),
    brightness_factor: float = Form(1.0),
    gaussian_blur_kernel: int = Form(0),
    gaussian_noise_sigma: float = Form(0.0),
    contrast_factor: float = Form(1.0),
    color_tint: Optional[str] = Form(None),
    color_tint_intensity: float = Form(0.3)
):
    content = await file.read()
    orig_bgr = decode_image_bytes(content)

    config = ImageDegradationConfig(
        brightness_factor=brightness_factor,
        gaussian_blur_kernel=gaussian_blur_kernel,
        gaussian_noise_sigma=gaussian_noise_sigma,
        contrast_factor=contrast_factor,
        color_tint=color_tint,
        color_tint_intensity=color_tint_intensity
    )

    degraded_bgr, degradation_summary = apply_image_degradation(orig_bgr, config)

    # Analyze degraded
    metrics = analyze_image_quality(degraded_bgr)
    classifier = get_classifier()
    predictions = classifier.predict(degraded_bgr, top_k=5)

    engine = get_monitoring_engine()
    res = engine.process_image_frame(metrics, predictions, is_stream=False)
    baseline = BaselineManager.get_or_create_default_image_baseline()

    # Also analyze clean original for side-by-side comparison
    orig_metrics = analyze_image_quality(orig_bgr)
    orig_preds = classifier.predict(orig_bgr, top_k=3)
    orig_baseline = baseline.get("stats", {}) if baseline else {}
    orig_rel, orig_risk, orig_st, orig_warns, _ = ReliabilityScoringEngine.evaluate_image(
        orig_metrics, orig_preds, orig_baseline
    )

    return {
        "degradation_applied": degradation_summary,
        "is_synthetically_degraded": (degradation_summary != "None (Clean Original)"),
        "degraded_image_base64": f"data:image/jpeg;base64,{encode_image_base64(degraded_bgr)}",
        "original_image_base64": f"data:image/jpeg;base64,{encode_image_base64(orig_bgr)}",
        "degraded_analysis": {
            "metrics": metrics,
            "predictions": predictions,
            "baseline_id": baseline["id"] if baseline else None,
            "baseline_deviations": res["deviations"],
            "reliability_score": res["reliability_score"],
            "risk_score": res["risk_score"],
            "status": res["status"],
            "warnings": res["warnings"]
        },
        "original_analysis": {
            "metrics": orig_metrics,
            "predictions": orig_preds,
            "reliability_score": orig_rel,
            "risk_score": orig_risk,
            "status": orig_st,
            "warnings": orig_warns
        }
    }

@api_router.get("/image/sample")
async def get_sample_image():
    """Generates a clean synthetic calibration test pattern."""
    img = np.full((320, 320, 3), 180, dtype=np.uint8)
    cv2.circle(img, (160, 160), 100, (45, 145, 230), -1) # orange circle
    cv2.circle(img, (160, 160), 50, (60, 180, 80), -1)   # green circle
    cv2.rectangle(img, (40, 40), (100, 100), (220, 70, 50), -1)
    cv2.putText(img, "RELIABILITY TEST", (45, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

    _, buffer = cv2.imencode(".jpg", img)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

# ==================== AUDIO SUBSYSTEM ====================

@api_router.post("/audio/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    try:
        samples, sr = parse_audio_data(content, filename=file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Audio decoding failure for '{file.filename}': {str(e)}"
        )

    # 1. Acoustic measurements
    metrics = analyze_audio_quality(samples, sr)

    # 2. CNN Acoustic Condition classification
    classifier = get_audio_classifier()
    classification = classifier.classify(
        samples, sr,
        peak_amp=metrics.peak_amplitude,
        clipping_pct=metrics.clipping_pct,
        rms_dbfs=metrics.rms_dbfs,
        snr_db=metrics.snr_db,
        spectral_centroid=metrics.spectral_centroid_hz
    )

    # 3. Unified engine evaluation
    engine = get_monitoring_engine()
    res = engine.process_audio_chunk(metrics, classification, is_stream=False)
    baseline = BaselineManager.get_or_create_default_audio_baseline()

    return AudioAnalysisResponse(
        metrics=metrics,
        classification=classification,
        baseline_id=baseline["id"] if baseline else None,
        baseline_deviations=res["deviations"],
        reliability_score=res["reliability_score"],
        risk_score=res["risk_score"],
        status=res["status"],
        warnings=res["warnings"],
        is_synthetically_degraded=False
    )

@api_router.post("/audio/degrade")
async def degrade_and_analyze_audio(
    file: UploadFile = File(...),
    volume_factor: float = Form(1.0),
    noise_type: Optional[str] = Form(None),
    noise_level: float = Form(0.0),
    clipping_gain: float = Form(1.0)
):
    content = await file.read()
    try:
        orig_samples, sr = parse_audio_data(content, filename=file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Audio decoding failure for '{file.filename}': {str(e)}"
        )

    config = AudioDegradationConfig(
        volume_factor=volume_factor,
        noise_type=noise_type,
        noise_level=noise_level,
        clipping_gain=clipping_gain
    )

    degraded_samples, summary = apply_audio_degradation(orig_samples, sr, config)

    # Metrics
    metrics = analyze_audio_quality(degraded_samples, sr)
    classifier = get_audio_classifier()
    classification = classifier.classify(
        degraded_samples, sr,
        peak_amp=metrics.peak_amplitude,
        clipping_pct=metrics.clipping_pct,
        rms_dbfs=metrics.rms_dbfs,
        snr_db=metrics.snr_db,
        spectral_centroid=metrics.spectral_centroid_hz
    )

    engine = get_monitoring_engine()
    res = engine.process_audio_chunk(metrics, classification, is_stream=False)
    baseline = BaselineManager.get_or_create_default_audio_baseline()

    # Original analysis
    orig_metrics = analyze_audio_quality(orig_samples, sr)
    orig_class = classifier.classify(orig_samples, sr, peak_amp=orig_metrics.peak_amplitude,
                                     clipping_pct=orig_metrics.clipping_pct, rms_dbfs=orig_metrics.rms_dbfs,
                                     snr_db=orig_metrics.snr_db, spectral_centroid=orig_metrics.spectral_centroid_hz)
    orig_b_stats = baseline.get("stats", {}) if baseline else {}
    orig_rel, orig_risk, orig_st, orig_warns, _ = ReliabilityScoringEngine.evaluate_audio(
        orig_metrics, orig_class, orig_b_stats
    )

    degraded_wav = samples_to_wav_bytes(degraded_samples, sr)
    orig_wav = samples_to_wav_bytes(orig_samples, sr)

    return {
        "degradation_applied": summary,
        "is_synthetically_degraded": (summary != "None (Clean Original)"),
        "degraded_audio_base64": f"data:audio/wav;base64,{base64.b64encode(degraded_wav).decode('utf-8')}",
        "original_audio_base64": f"data:audio/wav;base64,{base64.b64encode(orig_wav).decode('utf-8')}",
        "degraded_analysis": {
            "metrics": metrics,
            "classification": classification,
            "reliability_score": res["reliability_score"],
            "risk_score": res["risk_score"],
            "status": res["status"],
            "warnings": res["warnings"]
        },
        "original_analysis": {
            "metrics": orig_metrics,
            "classification": orig_class,
            "reliability_score": orig_rel,
            "risk_score": orig_risk,
            "status": orig_st,
            "warnings": orig_warns
        }
    }

@api_router.get("/audio/sample")
async def get_sample_audio():
    """Generates a clean synthetic 2-second harmonic test tone simulating speech fundamental formants."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Speech-like formant harmonics: 220Hz (fundamental) + 440Hz + 880Hz + 1760Hz with envelope
    f0 = 220.0
    sig = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * f0 * 2 * t) + 0.15 * np.sin(2 * np.pi * f0 * 4 * t)
    # Envelope (fade in and fade out)
    envelope = np.sin(np.pi * t / duration)
    sig = (sig * envelope).astype(np.float32)

    wav_bytes = samples_to_wav_bytes(sig, sr)
    return Response(content=wav_bytes, media_type="audio/wav")

# ==================== LOCAL LLM SUBSYSTEM ====================

@api_router.get("/llm/providers")
async def list_providers():
    registry = get_provider_registry()
    providers = registry.list_available_providers()
    return {
        "providers": providers,
        "active_provider": registry.get_provider().get_provider_name()
    }

@api_router.get("/llm/health")
async def check_llm_provider_health(provider: Optional[str] = None):
    registry = get_provider_registry()
    prov = registry.get_provider(provider)
    is_ok, msg = await prov.check_health()
    return {
        "provider": prov.get_provider_name(),
        "is_connected": is_ok,
        "message": msg
    }

@api_router.get("/llm/models")
async def list_llm_models(provider: Optional[str] = None):
    registry = get_provider_registry()
    prov = registry.get_provider(provider)
    models = await prov.list_models()
    return {
        "provider": prov.get_provider_name(),
        "models": models
    }

@api_router.post("/llm/test-connection", response_model=LLMDiagnosticReport)
async def test_llm_connection_diagnostic(req: LLMConnectionTestRequest):
    steps: List[DiagnosticStepResult] = []
    discovered_models: List[str] = []
    sample_response: Optional[str] = None
    saved_as_active = False

    # Mock provider fast-path
    if req.provider == "mock":
        steps.append(DiagnosticStepResult(step_number=1, name="Address & Port Validation", status="pass", latency_ms=0.1, message="Internal synthetic offline provider validated."))
        steps.append(DiagnosticStepResult(step_number=2, name="Server Reachability Probe", status="pass", latency_ms=0.1, message="In-process mock server is always reachable."))
        steps.append(DiagnosticStepResult(step_number=3, name="API Compatibility", status="pass", latency_ms=0.1, message="Native Python mock implementation active."))
        steps.append(DiagnosticStepResult(step_number=4, name="Model Discovery", status="pass", latency_ms=0.1, message="Discovered 3 virtual mock models: mock-phi-3, mock-llama-3, mock-mistral-7b."))
        steps.append(DiagnosticStepResult(step_number=5, name="Test Inference", status="pass", latency_ms=1.2, message="Synthetic test inference verified."))
        if req.save_if_successful:
            set_setting("llm_provider", "mock")
            get_provider_registry().set_active_provider("mock")
            saved_as_active = True
        return LLMDiagnosticReport(
            provider="mock",
            target_url="internal://mock",
            host="127.0.0.1",
            port=0,
            overall_status="connected",
            steps=steps,
            discovered_models=["mock-phi-3", "mock-llama-3", "mock-mistral-7b"],
            sample_response="Mock generation operational.",
            saved_as_active=saved_as_active
        )

    # 1. Determine raw target address
    if req.base_url:
        raw_addr = req.base_url
    elif req.host and req.port:
        raw_addr = f"http://{req.host}:{req.port}"
    elif req.host:
        raw_addr = f"http://{req.host}"
    else:
        settings = get_all_settings()
        if req.provider == "ollama":
            raw_addr = settings.get("ollama_url", "http://127.0.0.1:11434")
        elif req.provider == "lm_studio":
            raw_addr = settings.get("openai_compat_url", "http://127.0.0.1:1234/v1")
        elif req.provider == "llama_cpp":
            raw_addr = settings.get("llama_cpp_url", "http://127.0.0.1:8080/v1")
        elif req.provider == "custom_http":
            raw_addr = settings.get("custom_llm_url", "http://127.0.0.1:3000")
        else:
            raw_addr = "http://127.0.0.1:11434"

    # Step 1: Address and Port Validation
    t0 = time.time()
    def_port = 11434 if req.provider == "ollama" else 1234 if req.provider == "lm_studio" else 8080 if req.provider == "llama_cpp" else 3000
    is_valid, canon_url, host, err_msg, port = parse_and_validate_local_endpoint(
        raw_addr, default_port=def_port, allow_remote=req.allow_remote
    )
    s1_lat = round((time.time() - t0) * 1000.0, 2)

    if not is_valid:
        steps.append(DiagnosticStepResult(
            step_number=1,
            name="Address & Port Validation",
            status="fail",
            latency_ms=s1_lat,
            message=err_msg,
            remedy="Check host and port syntax. For security, only loopback addresses (127.0.0.1, localhost) are permitted unless 'allow_remote' is enabled."
        ))
        return LLMDiagnosticReport(
            provider=req.provider,
            target_url=raw_addr,
            host=host or "unknown",
            port=port,
            overall_status="failed",
            steps=steps,
            discovered_models=[]
        )

    steps.append(DiagnosticStepResult(
        step_number=1,
        name="Address & Port Validation",
        status="pass",
        latency_ms=s1_lat,
        message=f"Canonical URL '{canon_url}' validated (Host: {host}, Port: {port}). SSRF check passed."
    ))

    # Step 2: Server Reachability Probe
    t0 = time.time()
    probe_content_type = ""
    probe_body = ""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            probe_res = await client.get(canon_url)
            probe_content_type = probe_res.headers.get("content-type", "").lower()
            probe_body = probe_res.text[:300].strip().lower()
            s2_lat = round((time.time() - t0) * 1000.0, 2)
            steps.append(DiagnosticStepResult(
                step_number=2,
                name="Server Reachability Probe",
                status="pass",
                latency_ms=s2_lat,
                message=f"Host responded with HTTP {probe_res.status_code} in {s2_lat}ms."
            ))
    except httpx.ConnectError:
        s2_lat = round((time.time() - t0) * 1000.0, 2)
        remedy_msg = (
            f"Ollama server is not running on port {port}. Open terminal and run 'ollama serve'."
            if req.provider == "ollama" else
            f"Local server is not running on port {port}. Ensure {req.provider} is running and listening on port {port}."
        )
        steps.append(DiagnosticStepResult(
            step_number=2,
            name="Server Reachability Probe",
            status="fail",
            latency_ms=s2_lat,
            message=f"Connection refused at {canon_url}. No service is listening on port {port}.",
            remedy=remedy_msg
        ))
        return LLMDiagnosticReport(
            provider=req.provider,
            target_url=canon_url,
            host=host,
            port=port,
            overall_status="failed",
            steps=steps,
            discovered_models=[]
        )
    except httpx.TimeoutException:
        s2_lat = round((time.time() - t0) * 1000.0, 2)
        steps.append(DiagnosticStepResult(
            step_number=2,
            name="Server Reachability Probe",
            status="fail",
            latency_ms=s2_lat,
            message=f"Connection timed out after 3.0 seconds at {canon_url}.",
            remedy="Check whether a local firewall is blocking the port or the host is unresponsive."
        ))
        return LLMDiagnosticReport(
            provider=req.provider,
            target_url=canon_url,
            host=host,
            port=port,
            overall_status="failed",
            steps=steps,
            discovered_models=[]
        )
    except Exception as e:
        s2_lat = round((time.time() - t0) * 1000.0, 2)
        steps.append(DiagnosticStepResult(
            step_number=2,
            name="Server Reachability Probe",
            status="fail",
            latency_ms=s2_lat,
            message=f"Network error: {str(e)}",
            remedy="Verify endpoint reachability and network adapters."
        ))
        return LLMDiagnosticReport(
            provider=req.provider,
            target_url=canon_url,
            host=host,
            port=port,
            overall_status="failed",
            steps=steps,
            discovered_models=[]
        )

    # Step 3: API Compatibility Check & Step 4: Model Discovery
    t0 = time.time()
    is_frontend_html = (
        "text/html" in probe_content_type or
        probe_body.startswith("<!doctype") or
        probe_body.startswith("<html")
    )

    if is_frontend_html:
        s3_lat = round((time.time() - t0) * 1000.0, 2)
        steps.append(DiagnosticStepResult(
            step_number=3,
            name="API Compatibility",
            status="fail",
            latency_ms=s3_lat,
            message=(
                f"The server at {canon_url} is serving frontend HTML ('text/html'), not an LLM inference API. "
                "It appears to be a web/frontend development server. To connect to an LLM, point to an inference daemon "
                "(e.g., Ollama on port 11434 or LM Studio on port 1234) or configure an LLM proxy."
            ),
            remedy=(
                f"Port {port} is serving the web application UI. A frontend development server cannot directly process LLM inferences. "
                "Point the address to your inference daemon (e.g. http://localhost:11434 for Ollama or http://localhost:1234 for LM Studio), "
                "or run an LLM reverse proxy."
            )
        ))
        return LLMDiagnosticReport(
            provider=req.provider, target_url=canon_url, host=host, port=port,
            overall_status="failed", steps=steps, discovered_models=[]
        )

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            if req.provider == "ollama":
                tags_url = join_url_path(canon_url, "/api/tags")
                res = await client.get(tags_url)
                s3_lat = round((time.time() - t0) * 1000.0, 2)
                ct = res.headers.get("content-type", "").lower()
                if "text/html" in ct or res.text.strip().lower().startswith("<!doctype"):
                    steps.append(DiagnosticStepResult(
                        step_number=3,
                        name="API Compatibility",
                        status="fail",
                        latency_ms=s3_lat,
                        message=f"Endpoint {tags_url} returned HTML ('text/html') instead of JSON. Server is not an Ollama daemon.",
                        remedy="Confirm this is an Ollama daemon listening on this port."
                    ))
                    return LLMDiagnosticReport(
                        provider=req.provider, target_url=canon_url, host=host, port=port,
                        overall_status="failed", steps=steps, discovered_models=[]
                    )
                if res.status_code == 200:
                    steps.append(DiagnosticStepResult(
                        step_number=3,
                        name="API Compatibility",
                        status="pass",
                        latency_ms=s3_lat,
                        message=f"Ollama native API endpoint {tags_url} verified."
                    ))
                    try:
                        data = res.json()
                        discovered_models = [m.get("name", "unknown") for m in data.get("models", [])]
                    except Exception:
                        discovered_models = []
                else:
                    steps.append(DiagnosticStepResult(
                        step_number=3,
                        name="API Compatibility",
                        status="fail",
                        latency_ms=s3_lat,
                        message=f"Endpoint {tags_url} returned HTTP {res.status_code}: {res.text[:120]}",
                        remedy="Confirm this is an Ollama daemon. If running LM Studio or llama.cpp, select OpenAI-compatible provider."
                    ))
                    return LLMDiagnosticReport(
                        provider=req.provider, target_url=canon_url, host=host, port=port,
                        overall_status="failed", steps=steps, discovered_models=[]
                    )

            elif req.provider in ["lm_studio", "llama_cpp"]:
                models_url = join_url_path(canon_url, "/models")
                if not canon_url.endswith("/v1") and "/v1" not in canon_url:
                    models_url = join_url_path(canon_url, "/v1/models")
                res = await client.get(models_url)
                if res.status_code == 404:
                    alt_url = join_url_path(canon_url, "/models")
                    res = await client.get(alt_url)
                    if res.status_code == 200:
                        models_url = alt_url

                s3_lat = round((time.time() - t0) * 1000.0, 2)
                ct = res.headers.get("content-type", "").lower()
                if "text/html" in ct or res.text.strip().lower().startswith("<!doctype"):
                    steps.append(DiagnosticStepResult(
                        step_number=3,
                        name="API Compatibility",
                        status="fail",
                        latency_ms=s3_lat,
                        message=f"Endpoint {models_url} returned HTML ('text/html') instead of JSON.",
                        remedy="Check server path. Many servers require a '/v1' prefix like 'http://127.0.0.1:1234/v1'."
                    ))
                    return LLMDiagnosticReport(
                        provider=req.provider, target_url=canon_url, host=host, port=port,
                        overall_status="failed", steps=steps, discovered_models=[]
                    )
                if res.status_code == 200:
                    steps.append(DiagnosticStepResult(
                        step_number=3,
                        name="API Compatibility",
                        status="pass",
                        latency_ms=s3_lat,
                        message=f"OpenAI-compatible models endpoint {models_url} verified."
                    ))
                    try:
                        data = res.json()
                        discovered_models = [m.get("id", "unknown") for m in data.get("data", [])]
                    except Exception:
                        discovered_models = []
                else:
                    steps.append(DiagnosticStepResult(
                        step_number=3,
                        name="API Compatibility",
                        status="fail",
                        latency_ms=s3_lat,
                        message=f"Endpoint {models_url} returned HTTP {res.status_code}: {res.text[:120]}",
                        remedy="Check server path. Many servers require a '/v1' prefix like 'http://127.0.0.1:1234/v1'."
                    ))
                    return LLMDiagnosticReport(
                        provider=req.provider, target_url=canon_url, host=host, port=port,
                        overall_status="failed", steps=steps, discovered_models=[]
                    )

            elif req.provider == "custom_http":
                custom_prov = CustomLocalProvider(
                    base_url=canon_url,
                    api_path=req.api_path or "",
                    adapter_type=req.adapter_type or "auto",
                    fallback_model=req.model or "custom-model"
                )
                is_ok, health_msg = await custom_prov.check_health()
                s3_lat = round((time.time() - t0) * 1000.0, 2)
                if not is_ok:
                    steps.append(DiagnosticStepResult(
                        step_number=3,
                        name="API Compatibility",
                        status="fail",
                        latency_ms=s3_lat,
                        message=health_msg,
                        remedy="Verify custom HTTP server address, base path, and adapter configuration."
                    ))
                    return LLMDiagnosticReport(
                        provider=req.provider, target_url=canon_url, host=host, port=port,
                        overall_status="failed", steps=steps, discovered_models=[]
                    )
                steps.append(DiagnosticStepResult(
                    step_number=3,
                    name="API Compatibility",
                    status="pass",
                    latency_ms=s3_lat,
                    message=health_msg
                ))
                discovered_models = await custom_prov.list_models()

    except Exception as e:
        s3_lat = round((time.time() - t0) * 1000.0, 2)
        steps.append(DiagnosticStepResult(
            step_number=3,
            name="API Compatibility",
            status="fail",
            latency_ms=s3_lat,
            message=f"API check exception: {str(e)}",
            remedy="Verify server API specification."
        ))
        return LLMDiagnosticReport(
            provider=req.provider, target_url=canon_url, host=host, port=port,
            overall_status="failed", steps=steps, discovered_models=[]
        )

    # Step 4: Model Discovery validation
    if len(discovered_models) == 0:
        if req.model:
            discovered_models = [req.model]
            steps.append(DiagnosticStepResult(
                step_number=4,
                name="Model Discovery",
                status="pass",
                latency_ms=0.5,
                message=f"Model discovery not exposed by endpoint; utilizing specified model identifier '{req.model}'."
            ))
        else:
            steps.append(DiagnosticStepResult(
                step_number=4,
                name="Model Discovery",
                status="warn",
                latency_ms=0.5,
                message="Server connected successfully, but 0 models were discovered.",
                remedy=(
                    "Pull a local model in terminal: 'ollama pull llama3' or 'ollama pull mistral'."
                    if req.provider == "ollama" else
                    "Load a model into memory in LM Studio, or specify a manual model identifier in settings."
                )
            ))
    else:
        steps.append(DiagnosticStepResult(
            step_number=4,
            name="Model Discovery",
            status="pass",
            latency_ms=0.5,
            message=f"Discovered {len(discovered_models)} model(s): {', '.join(discovered_models[:4])}"
        ))

    # Step 5: Test Inference
    t0 = time.time()
    test_model = req.model if (req.model and (req.model in discovered_models or req.provider == "custom_http")) else (discovered_models[0] if discovered_models else None)

    if not test_model:
        steps.append(DiagnosticStepResult(
            step_number=5,
            name="Test Inference",
            status="warn",
            latency_ms=0.0,
            message="Test inference skipped because no active model is loaded or specified.",
            remedy="Install or load a model, or provide a manual model identifier."
        ))
    else:
        try:
            if req.provider == "custom_http":
                custom_prov = CustomLocalProvider(
                    base_url=canon_url,
                    api_path=req.api_path or "",
                    adapter_type=req.adapter_type or "auto",
                    fallback_model=test_model
                )
                res_obj = await custom_prov.generate(
                    prompt="Say READY",
                    model=test_model,
                    max_tokens=8
                )
                s5_lat = res_obj.duration_ms
                sample_response = res_obj.response_text.strip()
                steps.append(DiagnosticStepResult(
                    step_number=5,
                    name="Test Inference",
                    status="pass",
                    latency_ms=s5_lat,
                    message=f"Inference succeeded on '{test_model}' in {s5_lat}ms: '{sample_response}'"
                ))
            elif req.provider == "ollama":
                async with httpx.AsyncClient(timeout=10.0) as client:
                    gen_url = join_url_path(canon_url, "/api/generate")
                    gen_res = await client.post(gen_url, json={
                        "model": test_model,
                        "prompt": "Say READY",
                        "stream": False,
                        "options": {"num_predict": 8, "temperature": 0.0}
                    })
                    s5_lat = round((time.time() - t0) * 1000.0, 2)
                    if gen_res.status_code == 200:
                        sample_response = gen_res.json().get("response", "").strip()
                        steps.append(DiagnosticStepResult(
                            step_number=5,
                            name="Test Inference",
                            status="pass",
                            latency_ms=s5_lat,
                            message=f"Inference succeeded on '{test_model}' in {s5_lat}ms: '{sample_response}'"
                        ))
                    else:
                        steps.append(DiagnosticStepResult(
                            step_number=5,
                            name="Test Inference",
                            status="warn",
                            latency_ms=s5_lat,
                            message=f"Inference call returned HTTP {gen_res.status_code}: {gen_res.text[:100]}",
                            remedy="Model may need more GPU memory or is still loading."
                        ))
            else: # OpenAI
                async with httpx.AsyncClient(timeout=10.0) as client:
                    chat_url = join_url_path(canon_url, "/chat/completions")
                    if not canon_url.endswith("/v1") and "/v1" not in canon_url:
                        chat_url = join_url_path(canon_url, "/v1/chat/completions")
                    gen_res = await client.post(chat_url, json={
                        "model": test_model,
                        "messages": [{"role": "user", "content": "Say READY"}],
                        "max_tokens": 8,
                        "temperature": 0.0
                    })
                    if gen_res.status_code == 404:
                        chat_url = join_url_path(canon_url, "/chat/completions")
                        gen_res = await client.post(chat_url, json={
                            "model": test_model,
                            "messages": [{"role": "user", "content": "Say READY"}],
                            "max_tokens": 8,
                            "temperature": 0.0
                        })
                    s5_lat = round((time.time() - t0) * 1000.0, 2)
                    if gen_res.status_code == 200:
                        data = gen_res.json()
                        sample_response = data["choices"][0]["message"]["content"].strip()
                        steps.append(DiagnosticStepResult(
                            step_number=5,
                            name="Test Inference",
                            status="pass",
                            latency_ms=s5_lat,
                            message=f"Inference succeeded on '{test_model}' in {s5_lat}ms: '{sample_response}'"
                        ))
                    else:
                        steps.append(DiagnosticStepResult(
                            step_number=5,
                            name="Test Inference",
                            status="warn",
                            latency_ms=s5_lat,
                            message=f"Inference call returned HTTP {gen_res.status_code}: {gen_res.text[:100]}",
                            remedy="Check server logs and GPU memory allocation."
                        ))
        except Exception as e:
            s5_lat = round((time.time() - t0) * 1000.0, 2)
            steps.append(DiagnosticStepResult(
                step_number=5,
                name="Test Inference",
                status="warn",
                latency_ms=s5_lat,
                message=f"Inference test warning: {str(e)}",
                remedy="Model response timed out or was interrupted. Server remains reachable."
            ))

    has_fail = any(s.status == "fail" for s in steps)
    has_warn = any(s.status == "warn" for s in steps)
    overall_status = "failed" if has_fail else ("warning" if has_warn else "connected")

    # Save to database and registry if requested and successfully connected
    if req.save_if_successful and overall_status in ["connected", "warning"]:
        if req.provider == "ollama":
            set_setting("ollama_url", canon_url)
        elif req.provider == "lm_studio":
            set_setting("openai_compat_url", canon_url)
        elif req.provider == "llama_cpp":
            set_setting("llama_cpp_url", canon_url)
        elif req.provider == "custom_http":
            set_setting("custom_llm_url", canon_url)
            if req.api_path is not None:
                set_setting("custom_llm_path", req.api_path)
            if req.adapter_type:
                set_setting("custom_llm_type", req.adapter_type)
            if req.model:
                set_setting("custom_llm_model", req.model)
        set_setting("llm_provider", req.provider)

        reg = get_provider_registry()
        reg.update_provider_url(req.provider, canon_url)
        if req.provider == "custom_http":
            reg.update_custom_provider_config(
                base_url=canon_url,
                api_path=req.api_path,
                adapter_type=req.adapter_type,
                fallback_model=req.model
            )
        reg.set_active_provider(req.provider)
        reg.sync_from_settings()
        saved_as_active = True

    return LLMDiagnosticReport(
        provider=req.provider,
        target_url=canon_url,
        host=host,
        port=port,
        overall_status=overall_status,
        steps=steps,
        discovered_models=discovered_models,
        sample_response=sample_response,
        saved_as_active=saved_as_active
    )

@api_router.post("/llm/generate", response_model=LLMResponseInfo)
async def generate_llm_response(req: LLMGenerateRequest):
    registry = get_provider_registry()
    prov = registry.get_provider(req.provider)

    try:
        provider_resp = await prov.generate(
            prompt=req.prompt,
            model=req.model,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM Generation Error: {str(e)}")

    # Response reliability monitoring
    resp_risk, warnings, is_uncert, is_empty = ResponseMonitor.evaluate_response(
        prompt=req.prompt,
        response_text=provider_resp.response_text,
        input_risk_score=req.current_input_risk_score or 0.0,
        input_subsystem=req.current_input_subsystem
    )

    interaction_id = ResponseMonitor.record_interaction(
        provider=req.provider,
        model=req.model,
        prompt=req.prompt,
        response=provider_resp.response_text,
        input_subsystem=req.current_input_subsystem,
        input_risk_score=req.current_input_risk_score or 0.0,
        response_risk_score=resp_risk,
        warnings=warnings
    )

    return LLMResponseInfo(
        provider=req.provider,
        model=req.model,
        prompt=req.prompt,
        response=provider_resp.response_text,
        duration_ms=provider_resp.duration_ms,
        input_risk_score=req.current_input_risk_score or 0.0,
        response_risk_score=resp_risk,
        warnings=warnings,
        is_uncertain=is_uncert,
        is_empty=is_empty,
        interaction_id=interaction_id
    )

@api_router.post("/llm/generate/stream")
async def generate_llm_stream(req: LLMGenerateRequest):
    registry = get_provider_registry()
    prov = registry.get_provider(req.provider)

    async def token_generator():
        async for chunk in prov.generate_stream(
            prompt=req.prompt,
            model=req.model,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        ):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@api_router.post("/llm/feedback")
async def submit_llm_feedback(fb: LLMFeedbackRequest):
    ResponseMonitor.record_feedback(fb.interaction_id, fb.user_rating, fb.user_notes)
    return {"status": "feedback_recorded", "interaction_id": fb.interaction_id}

# ==================== BASELINE PROFILES ====================

@api_router.get("/baselines", response_model=List[BaselineInfo])
async def list_baselines_endpoint(type: Optional[str] = None):
    # Ensure default baselines exist
    BaselineManager.get_or_create_default_image_baseline()
    BaselineManager.get_or_create_default_audio_baseline()
    return get_baselines(type)

@api_router.get("/baselines/active")
async def get_active_baseline_endpoint(type: str = Query(..., pattern="^(image|audio)$")):
    if type == "image":
        return BaselineManager.get_or_create_default_image_baseline()
    else:
        return BaselineManager.get_or_create_default_audio_baseline()

@api_router.post("/baselines/create/image")
async def create_image_baseline_endpoint(
    name: str = Form(...),
    description: str = Form(""),
    files: List[UploadFile] = File(...)
):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required to calibrate a baseline.")

    metrics_list = []
    preds_list = []
    classifier = get_classifier()

    for f in files:
        b = await f.read()
        img = decode_image_bytes(b)
        m = analyze_image_quality(img)
        p = classifier.predict(img, top_k=5)
        metrics_list.append(m)
        preds_list.append(p)

    baseline = BaselineManager.calibrate_image_baseline(
        name=name,
        description=description,
        metrics_list=metrics_list,
        predictions_list=preds_list,
        set_active=True
    )
    return baseline

@api_router.post("/baselines/create/audio")
async def create_audio_baseline_endpoint(
    name: str = Form(...),
    description: str = Form(""),
    files: List[UploadFile] = File(...)
):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one audio file is required to calibrate a baseline.")

    metrics_list = []
    errors = []
    for f in files:
        b = await f.read()
        try:
            samples, sr = parse_audio_data(b, filename=f.filename)
            m = analyze_audio_quality(samples, sr)
            metrics_list.append(m)
        except Exception as e:
            errors.append(f"'{f.filename}': {str(e)}")

    if len(metrics_list) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Could not calibrate baseline. None of the {len(files)} files could be decoded. Errors: {'; '.join(errors)}"
        )

    baseline = BaselineManager.calibrate_audio_baseline(
        name=name,
        description=description,
        metrics_list=metrics_list,
        set_active=True
    )
    return baseline

@api_router.post("/baselines/activate/{baseline_id}")
async def activate_baseline_endpoint(baseline_id: str):
    ok = set_active_baseline(baseline_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Baseline not found.")
    
    # Notify engine based on baseline type
    all_b = get_baselines()
    target = next((b for b in all_b if b.id == baseline_id), None)
    if target:
        get_monitoring_engine().notify_baseline_updated(target.type)

    return {"status": "activated", "baseline_id": baseline_id}

@api_router.delete("/baselines/{baseline_id}")
async def delete_baseline_endpoint(baseline_id: str, confirm: bool = Query(False)):
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required to delete baseline profile.")
    delete_baseline(baseline_id)
    return {"status": "deleted", "baseline_id": baseline_id}

# ==================== DATASETS & LOCAL TRAINING ====================

@api_router.get("/datasets")
async def list_datasets_endpoint():
    DatasetManager.create_synthetic_demo_dataset()
    return DatasetManager.list_datasets()

@api_router.post("/datasets/demo")
async def create_demo_dataset_endpoint():
    return DatasetManager.create_synthetic_demo_dataset()

@api_router.get("/models/versions")
async def list_model_versions_endpoint():
    return ModelTrainingManager.list_model_versions()

@api_router.post("/models/activate/{version_id}")
async def activate_model_endpoint(version_id: str):
    return ModelTrainingManager.activate_model_version(version_id)

@api_router.post("/models/train")
async def run_training_endpoint(
    dataset_id: str = Form(...),
    epochs: int = Form(3),
    lr: float = Form(0.001)
):
    ds = DatasetManager.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    res = ModelTrainingManager.run_transfer_learning(
        dataset_path=ds["path"],
        classes=ds["classes"],
        epochs=epochs,
        lr=lr
    )
    return res

@api_router.get("/models/train/status")
async def get_training_status_endpoint():
    return ModelTrainingManager.get_training_status()

# ==================== EVENTS & SETTINGS ====================

@api_router.get("/events")
async def list_events_endpoint(
    subsystem: Optional[str] = Query("all"),
    severity: Optional[str] = Query("all"),
    limit: int = Query(50)
):
    return get_recent_events(limit=limit, subsystem=subsystem, severity=severity)

@api_router.delete("/events")
async def clear_events_endpoint(confirm: bool = Query(False)):
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required to clear event history.")
    clear_events()
    return {"status": "cleared"}

@api_router.post("/monitor/state")
async def set_monitor_active_endpoint(active: bool = Body(..., embed=True)):
    engine = get_monitoring_engine()
    engine.set_monitoring_active(active)
    return {
        "status": "updated",
        "monitoring_active": active,
        "overall": engine.get_system_overall_status()
    }

@api_router.get("/settings")
async def get_settings_endpoint():
    return get_all_settings()

@api_router.post("/settings")
async def update_settings_endpoint(settings: Dict[str, str] = Body(...)):
    for k, v in settings.items():
        set_setting(k, str(v))
    get_provider_registry().sync_from_settings()
    return {"status": "saved", "settings": get_all_settings()}
