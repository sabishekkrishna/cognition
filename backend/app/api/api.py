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
    res = engine.process_image_frame(metrics, predictions)

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
    res = engine.process_image_frame(metrics, predictions)
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
    samples, sr = parse_audio_data(content)

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
    res = engine.process_audio_chunk(metrics, classification)
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
    orig_samples, sr = parse_audio_data(content)

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
    res = engine.process_audio_chunk(metrics, classification)
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
    for f in files:
        b = await f.read()
        samples, sr = parse_audio_data(b)
        m = analyze_audio_quality(samples, sr)
        metrics_list.append(m)

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

@api_router.get("/settings")
async def get_settings_endpoint():
    return get_all_settings()

@api_router.post("/settings")
async def update_settings_endpoint(settings: Dict[str, str] = Body(...)):
    for k, v in settings.items():
        set_setting(k, str(v))
    return {"status": "saved", "settings": get_all_settings()}
