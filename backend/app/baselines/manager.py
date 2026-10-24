import numpy as np
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from ..storage.database import (
    save_baseline, get_baselines, get_active_baseline,
    set_active_baseline, delete_baseline
)
from ..schemas.schemas import ImageQualityMetrics, AudioQualityMetrics, ModelPredictions, BaselineInfo

# Domain-specific minimum variability thresholds to prevent division by zero or distortion on identical inputs
DOMAIN_MIN_MAD: Dict[str, float] = {
    # Audio metrics
    "spectral_centroid_hz": 100.0,
    "spectral_rolloff_hz": 150.0,
    "rms_dbfs": 2.0,
    "background_noise_db": 3.0,
    "snr_db": 2.5,
    "speech_ratio": 0.05,
    "silence_ratio": 0.05,
    "clipping_pct": 0.5,
    "peak_amplitude": 0.05,
    "duration_sec": 0.5,
    # Image metrics
    "brightness": 5.0,
    "sharpness": 10.0,
    "contrast": 4.0,
    "color_mean_r": 5.0,
    "color_mean_g": 5.0,
    "color_mean_b": 5.0,
    "color_cast_delta": 3.0,
    "noise_estimate": 1.0,
    "underexposure_pct": 1.0,
    "overexposure_pct": 1.0,
    "model_confidence": 0.05,
    "model_entropy": 0.2,
    "model_margin": 0.05
}

def compute_robust_distribution(values: List[float], metric_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes robust descriptive statistics for a distribution:
    mean, std, median, MAD (Median Absolute Deviation scaled by 1.4826),
    10th and 90th percentiles, 5th and 95th percentiles, min, max.
    Guarantees non-zero MAD using domain-specific fallback thresholds.
    """
    min_threshold = DOMAIN_MIN_MAD.get(metric_name, 1.0) if metric_name else 1.0

    if not values:
        return {
            "mean": 0.0, "std": 0.0, "median": 0.0, "mad": min_threshold,
            "p10": 0.0, "p90": 0.0, "p5": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0
        }

    arr = np.array(values, dtype=np.float64)
    med = float(np.median(arr))
    # Median Absolute Deviation (MAD), scaled by 1.4826 for normal consistency
    abs_deviations = np.abs(arr - med)
    raw_mad = float(np.median(abs_deviations) * 1.4826)
    
    # If MAD is near zero (e.g. identical samples), use std or domain-specific floor
    if raw_mad < 1e-4:
        sample_std = float(np.std(arr))
        mad = sample_std if sample_std > 1e-4 else min_threshold
    else:
        mad = max(raw_mad, min_threshold * 0.2) # Avoid unrealistically hyper-sensitive MAD

    return {
        "mean": float(round(np.mean(arr), 2)),
        "std": float(round(np.std(arr), 2)),
        "median": float(round(med, 2)),
        "mad": float(round(mad, 2)),
        "p10": float(round(np.percentile(arr, 10), 2)),
        "p90": float(round(np.percentile(arr, 90), 2)),
        "p5": float(round(np.percentile(arr, 5), 2)),
        "p95": float(round(np.percentile(arr, 95), 2)),
        "min": float(round(np.min(arr), 2)),
        "max": float(round(np.max(arr), 2))
    }

class BaselineManager:
    """
    Manages baseline calibration, persistence, and deviation comparison
    for image and audio profiles.
    """

    @staticmethod
    def calibrate_image_baseline(name: str,
                                description: str,
                                metrics_list: List[ImageQualityMetrics],
                                predictions_list: Optional[List[ModelPredictions]] = None,
                                set_active: bool = True) -> BaselineInfo:
        if not metrics_list:
            raise ValueError("At least one image is required to calibrate a baseline profile.")

        sample_count = len(metrics_list)
        stats: Dict[str, Any] = {}

        # Aggregate image metrics
        metric_keys = [
            "brightness", "sharpness", "contrast",
            "color_mean_r", "color_mean_g", "color_mean_b", "color_cast_delta",
            "noise_estimate", "underexposure_pct", "overexposure_pct"
        ]

        for k in metric_keys:
            vals = [getattr(m, k) for m in metrics_list]
            stats[k] = compute_robust_distribution(vals, metric_name=k)

        # Model confidence statistics & reference embedding
        if predictions_list:
            top_probs = [p.top_probability for p in predictions_list]
            entropies = [p.entropy for p in predictions_list]
            margins = [p.margin for p in predictions_list]

            stats["model_confidence"] = compute_robust_distribution(top_probs, metric_name="model_confidence")
            stats["model_entropy"] = compute_robust_distribution(entropies, metric_name="model_entropy")
            stats["model_margin"] = compute_robust_distribution(margins, metric_name="model_margin")

        stats["_schema_version"] = "v1.0"
        baseline_id = f"img_base_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        save_baseline(baseline_id, name, "image", description, sample_count, stats, set_active=set_active)

        if set_active:
            try:
                from ..monitoring.engine import get_monitoring_engine
                get_monitoring_engine().notify_baseline_updated("image")
            except Exception:
                pass

        return BaselineInfo(
            id=baseline_id,
            name=name,
            type="image",
            description=description,
            is_active=set_active,
            sample_count=sample_count,
            stats=stats,
            created_at=time.time(),
            updated_at=time.time()
        )

    @staticmethod
    def calibrate_audio_baseline(name: str,
                                description: str,
                                metrics_list: List[AudioQualityMetrics],
                                set_active: bool = True) -> BaselineInfo:
        if not metrics_list:
            raise ValueError("At least one audio recording is required to calibrate a baseline profile.")

        sample_count = len(metrics_list)
        stats: Dict[str, Any] = {}

        audio_keys = [
            "rms_dbfs", "peak_amplitude", "clipping_pct",
            "background_noise_db", "snr_db", "speech_ratio",
            "spectral_centroid_hz", "spectral_rolloff_hz"
        ]

        for k in audio_keys:
            vals = [getattr(m, k) for m in metrics_list]
            stats[k] = compute_robust_distribution(vals, metric_name=k)

        stats["_schema_version"] = "v1.0"
        baseline_id = f"aud_base_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        save_baseline(baseline_id, name, "audio", description, sample_count, stats, set_active=set_active)

        if set_active:
            try:
                from ..monitoring.engine import get_monitoring_engine
                get_monitoring_engine().notify_baseline_updated("audio")
            except Exception:
                pass

        return BaselineInfo(
            id=baseline_id,
            name=name,
            type="audio",
            description=description,
            is_active=set_active,
            sample_count=sample_count,
            stats=stats,
            created_at=time.time(),
            updated_at=time.time()
        )

    @staticmethod
    def compute_image_deviations(metrics: ImageQualityMetrics,
                                 baseline_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes normalized robust Z-scores / deviations:
        deviation = (value - median) / MAD
        """
        deviations: Dict[str, float] = {}
        for key in ["brightness", "sharpness", "contrast", "color_cast_delta", "noise_estimate", "underexposure_pct", "overexposure_pct"]:
            if key in baseline_stats:
                b = baseline_stats[key]
                median = b.get("median", 0.0)
                min_mad = DOMAIN_MIN_MAD.get(key, 1.0)
                mad = b.get("mad", min_mad)
                if mad < min_mad * 0.2:
                    mad = min_mad
                val = getattr(metrics, key, 0.0)
                dev = (val - median) / mad
                deviations[key] = float(round(dev, 2))
        return deviations

    @staticmethod
    def compute_audio_deviations(metrics: AudioQualityMetrics,
                                 baseline_stats: Dict[str, Any]) -> Dict[str, float]:
        deviations: Dict[str, float] = {}
        for key in ["rms_dbfs", "clipping_pct", "background_noise_db", "snr_db", "spectral_centroid_hz"]:
            if key in baseline_stats:
                b = baseline_stats[key]
                median = b.get("median", 0.0)
                min_mad = DOMAIN_MIN_MAD.get(key, 1.0)
                mad = b.get("mad", min_mad)
                if mad < min_mad * 0.2:
                    mad = min_mad
                val = getattr(metrics, key, 0.0)
                dev = (val - median) / mad
                deviations[key] = float(round(dev, 2))
        return deviations

    @staticmethod
    def get_or_create_default_image_baseline() -> Dict[str, Any]:
        """Returns active baseline or creates a standard clean-room default profile."""
        active = get_active_baseline("image")
        if active:
            return active

        # Standard clean reference profile based on standard photographic lighting conditions
        default_stats = {
            "brightness": {"median": 128.0, "mad": 30.0, "p5": 70.0, "p95": 190.0, "min": 30.0, "max": 230.0},
            "sharpness": {"median": 180.0, "mad": 60.0, "p5": 80.0, "p95": 450.0, "min": 50.0, "max": 800.0},
            "contrast": {"median": 55.0, "mad": 15.0, "p5": 25.0, "p95": 85.0, "min": 15.0, "max": 100.0},
            "color_cast_delta": {"median": 12.0, "mad": 8.0, "p5": 2.0, "p95": 30.0, "min": 0.0, "max": 45.0},
            "noise_estimate": {"median": 4.5, "mad": 2.0, "p5": 1.0, "p95": 10.0, "min": 0.5, "max": 15.0},
            "underexposure_pct": {"median": 0.5, "mad": 0.5, "p5": 0.0, "p95": 4.0, "min": 0.0, "max": 8.0},
            "overexposure_pct": {"median": 0.5, "mad": 0.5, "p5": 0.0, "p95": 3.0, "min": 0.0, "max": 6.0},
            "model_entropy": {"median": 2.2, "mad": 0.8, "p5": 0.5, "p95": 4.5, "min": 0.1, "max": 6.0},
            "model_confidence": {"median": 0.75, "mad": 0.15, "p5": 0.35, "p95": 0.98, "min": 0.15, "max": 1.0}
        }
        baseline_id = "img_base_default"
        save_baseline(baseline_id, "Standard Clear Photographic Baseline", "image",
                      "Default factory calibration for clean, well-lit environments.", 50, default_stats, set_active=True)
        return get_active_baseline("image")

    @staticmethod
    def get_or_create_default_audio_baseline() -> Dict[str, Any]:
        """Returns active audio baseline or creates a standard office/quiet room default profile."""
        active = get_active_baseline("audio")
        if active:
            return active

        default_stats = {
            "rms_dbfs": {"median": -22.0, "mad": 5.0, "p5": -32.0, "p95": -14.0, "min": -45.0, "max": -6.0},
            "clipping_pct": {"median": 0.0, "mad": 0.1, "p5": 0.0, "p95": 0.2, "min": 0.0, "max": 0.5},
            "background_noise_db": {"median": -45.0, "mad": 6.0, "p5": -55.0, "p95": -35.0, "min": -65.0, "max": -28.0},
            "snr_db": {"median": 24.0, "mad": 5.0, "p5": 14.0, "p95": 36.0, "min": 8.0, "max": 45.0},
            "speech_ratio": {"median": 0.65, "mad": 0.15, "p5": 0.30, "p95": 0.90, "min": 0.10, "max": 0.98},
            "spectral_centroid_hz": {"median": 1850.0, "mad": 400.0, "p5": 900.0, "p95": 3200.0, "min": 400.0, "max": 4500.0}
        }
        baseline_id = "aud_base_default"
        save_baseline(baseline_id, "Quiet Studio / Desk Mic Baseline", "audio",
                      "Default factory calibration for speech recognition in low background noise.", 50, default_stats, set_active=True)
        return get_active_baseline("audio")
