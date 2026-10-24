from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Image Metrics
class ImageQualityMetrics(BaseModel):
    brightness: float = Field(..., description="Mean grayscale intensity [0-255]")
    sharpness: float = Field(..., description="Variance of Laplacian (blur indicator)")
    contrast: float = Field(..., description="Standard deviation of grayscale values")
    color_mean_r: float = Field(..., description="Mean red channel intensity [0-255]")
    color_mean_g: float = Field(..., description="Mean green channel intensity [0-255]")
    color_mean_b: float = Field(..., description="Mean blue channel intensity [0-255]")
    color_cast_delta: float = Field(..., description="Max channel deviation indicator")
    noise_estimate: float = Field(..., description="Estimated Gaussian noise sigma")
    underexposure_pct: float = Field(..., description="Percentage of near-black pixels (<10)")
    overexposure_pct: float = Field(..., description="Percentage of saturated pixels (>245)")
    width: int
    height: int
    resolution_mp: float

class PredictionItem(BaseModel):
    class_id: int
    class_name: str
    probability: float

class ModelPredictions(BaseModel):
    model_name: str
    top_predictions: List[PredictionItem]
    top_class: str
    top_probability: float
    entropy: float
    margin: float
    embedding_dimension: int
    embedding_sample: Optional[List[float]] = None

class MetricWarning(BaseModel):
    subsystem: str
    metric: str
    severity: str # 'caution', 'warning'
    measured_value: float
    expected_range: str
    explanation: str
    recommendation: str

class ImageAnalysisResponse(BaseModel):
    metrics: ImageQualityMetrics
    predictions: ModelPredictions
    baseline_id: Optional[str]
    baseline_deviations: Dict[str, float]
    reliability_score: float # 0 to 100
    risk_score: float        # 0 to 100
    status: str              # 'normal', 'caution', 'warning'
    warnings: List[MetricWarning]
    is_synthetically_degraded: bool = False
    degradation_type: Optional[str] = None

# Audio Metrics
class AudioQualityMetrics(BaseModel):
    rms_dbfs: float
    peak_amplitude: float
    clipping_pct: float
    background_noise_db: float
    snr_db: float
    speech_ratio: float
    silence_ratio: float
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    duration_sec: float
    sample_rate: int

class AudioClassificationResult(BaseModel):
    model_name: str
    primary_condition: str
    condition_probabilities: Dict[str, float]

class AudioAnalysisResponse(BaseModel):
    metrics: AudioQualityMetrics
    classification: AudioClassificationResult
    baseline_id: Optional[str]
    baseline_deviations: Dict[str, float]
    reliability_score: float
    risk_score: float
    status: str
    warnings: List[MetricWarning]
    is_synthetically_degraded: bool = False
    degradation_type: Optional[str] = None

# Synthetic Degradation Requests
class ImageDegradationConfig(BaseModel):
    brightness_factor: float = 1.0  # <1.0 darker, >1.0 brighter
    gaussian_blur_kernel: int = 0   # 0 = none, e.g. 5, 11, 21
    gaussian_noise_sigma: float = 0.0 # e.g. 15.0, 30.0
    contrast_factor: float = 1.0    # <1.0 lower contrast
    color_tint: Optional[str] = None # 'red', 'green', 'blue', 'yellow', None
    color_tint_intensity: float = 0.3

class AudioDegradationConfig(BaseModel):
    volume_factor: float = 1.0      # <1.0 quieter, >1.0 louder
    noise_type: Optional[str] = None # 'white', 'pink', 'low_rumble'
    noise_level: float = 0.0        # 0.0 to 1.0
    clipping_gain: float = 1.0      # >1.0 forces clipping

# Baseline Schemas
class BaselineCreateRequest(BaseModel):
    name: str
    type: str # 'image' or 'audio'
    description: Optional[str] = ""
    confirm_replace: bool = False

class BaselineInfo(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str]
    is_active: bool
    sample_count: int
    stats: Dict[str, Any]
    created_at: float
    updated_at: float

# LLM Schemas
class LLMGenerateRequest(BaseModel):
    provider: str = "ollama"
    model: str
    prompt: str
    system_prompt: Optional[str] = "You are an AI assistant. Answer concisely and factually."
    temperature: float = 0.7
    max_tokens: int = 512
    include_response_monitoring: bool = True
    current_input_risk_score: Optional[float] = 0.0
    current_input_subsystem: Optional[str] = None

class LLMResponseInfo(BaseModel):
    provider: str
    model: str
    prompt: str
    response: str
    duration_ms: float
    input_risk_score: float
    response_risk_score: float
    warnings: List[str]
    is_uncertain: bool
    is_empty: bool
    interaction_id: Optional[int] = None

class LLMFeedbackRequest(BaseModel):
    interaction_id: int
    user_rating: str # 'correct', 'incorrect', 'uncertain'
    user_notes: Optional[str] = None

class ProviderConfig(BaseModel):
    provider: str
    base_url: str
    timeout_sec: float = 30.0

# Live Monitoring Status
class SystemOverallStatus(BaseModel):
    overall_state: str # 'normal', 'caution', 'warning', 'unavailable'
    overall_reliability_score: float
    image_state: str
    image_reliability_score: float
    audio_state: str
    audio_reliability_score: float
    llm_state: str
    active_warnings: List[MetricWarning]
    recent_events_count: int
    timestamp: float
