from typing import List, Dict, Any, Tuple
from ..schemas.schemas import ImageQualityMetrics, AudioQualityMetrics, ModelPredictions, MetricWarning, AudioClassificationResult

class ReliabilityScoringEngine:
    """
    Computes explainable, modular Reliability and Risk scores [0 - 100]
    for visual and acoustic inputs compared against baseline distributions.

    Scoring Methodology:
    - Base reliability starts at 100.0.
    - Each detected metric deviation or anomaly accrues a calibrated penalty
      weighted by its documented impact on downstream perception models.
    - Physical signal degradation (e.g. extreme blur, near-total darkness, high clipping)
      carries primary weight, while model prediction entropy modulates the final score.
    - Risk Score = 100.0 - Reliability Score.
    """

    # Image feature penalty weights and tolerance thresholds
    IMAGE_WEIGHTS = {
        "sharpness": 30.0,      # Severe blur critically impairs edge & texture detection
        "brightness": 20.0,     # Extreme darkness/saturation impairs dynamic range
        "contrast": 15.0,       # Low contrast degrades feature discriminability
        "noise": 15.0,          # High ISO/sensor noise corrupts high-frequency activations
        "color_cast": 10.0,     # Severe color imbalance shifts color-sensitive features
        "model_entropy": 10.0   # High prediction entropy signals model uncertainty
    }

    # Audio feature penalty weights
    AUDIO_WEIGHTS = {
        "snr": 30.0,            # Low signal-to-noise ratio severely degrades ASR word error rate
        "clipping": 25.0,       # Non-linear distortion produces spurious harmonics
        "loudness": 20.0,       # Too quiet (< -40 dBFS) or too loud causes transcription failure
        "noise_floor": 15.0,    # High background noise masks quiet phonemes
        "condition": 10.0       # CNN classifier detecting non-speech or mechanical interference
    }

    @classmethod
    def evaluate_image(cls,
                       metrics: ImageQualityMetrics,
                       predictions: ModelPredictions,
                       baseline_stats: Dict[str, Any]) -> Tuple[float, float, str, List[MetricWarning], Dict[str, float]]:
        """
        Evaluates visual input reliability and produces structured, actionable warnings.
        Returns: (reliability_score, risk_score, status, warnings, deviations)
        """
        penalties = 0.0
        warnings: List[MetricWarning] = []
        deviations: Dict[str, float] = {}

        # 1. Sharpness Evaluation (Laplacian variance)
        sharp_base = baseline_stats.get("sharpness", {"median": 180.0, "mad": 60.0, "p5": 70.0, "p95": 500.0})
        sharp_median = sharp_base.get("median", 180.0)
        sharp_mad = sharp_base.get("mad", 60.0)
        sharp_p5 = sharp_base.get("p5", 70.0)
        sharp_dev = (metrics.sharpness - sharp_median) / (sharp_mad if sharp_mad > 0 else 1.0)
        deviations["sharpness"] = float(round(sharp_dev, 2))

        if metrics.sharpness < 40.0:
            penalties += cls.IMAGE_WEIGHTS["sharpness"] * 1.0
            warnings.append(MetricWarning(
                subsystem="image",
                metric="sharpness",
                severity="warning",
                measured_value=metrics.sharpness,
                expected_range=f"≥ {sharp_p5:.1f}",
                explanation="Image sharpness is critically degraded (severe blur detected). The visual model will likely fail to resolve boundaries or small objects.",
                recommendation="Steady the camera, clean the lens, or refocus before relying on this visual analysis."
            ))
        elif metrics.sharpness < sharp_p5 or sharp_dev < -2.0:
            penalties += cls.IMAGE_WEIGHTS["sharpness"] * 0.5
            warnings.append(MetricWarning(
                subsystem="image",
                metric="sharpness",
                severity="caution",
                measured_value=metrics.sharpness,
                expected_range=f"≥ {sharp_p5:.1f}",
                explanation="Sharpness is below calibrated baseline. Minor blur may soften fine textures.",
                recommendation="Check camera stabilization and lighting."
            ))

        # 2. Brightness & Exposure Evaluation
        bright_base = baseline_stats.get("brightness", {"median": 128.0, "mad": 30.0, "p5": 60.0, "p95": 195.0})
        bright_median = bright_base.get("median", 128.0)
        bright_mad = bright_base.get("mad", 30.0)
        bright_p5 = bright_base.get("p5", 60.0)
        bright_p95 = bright_base.get("p95", 195.0)
        bright_dev = (metrics.brightness - bright_median) / (bright_mad if bright_mad > 0 else 1.0)
        deviations["brightness"] = float(round(bright_dev, 2))

        if metrics.brightness < 40.0 or metrics.underexposure_pct > 25.0:
            penalties += cls.IMAGE_WEIGHTS["brightness"] * 1.0
            warnings.append(MetricWarning(
                subsystem="image",
                metric="brightness",
                severity="warning",
                measured_value=metrics.brightness,
                expected_range=f"{bright_p5:.0f} - {bright_p95:.0f}",
                explanation="Input is severely underexposed or in near-total darkness. Significant shadow clipping conceals scene details.",
                recommendation="Increase ambient lighting or enable camera exposure compensation."
            ))
        elif metrics.brightness > 220.0 or metrics.overexposure_pct > 20.0:
            penalties += cls.IMAGE_WEIGHTS["brightness"] * 1.0
            warnings.append(MetricWarning(
                subsystem="image",
                metric="brightness",
                severity="warning",
                measured_value=metrics.brightness,
                expected_range=f"{bright_p5:.0f} - {bright_p95:.0f}",
                explanation="Input is severely overexposed with washed-out highlights and blown pixel saturation.",
                recommendation="Reduce direct harsh lighting or angle the camera away from intense light sources."
            ))
        elif metrics.brightness < bright_p5 or metrics.brightness > bright_p95:
            penalties += cls.IMAGE_WEIGHTS["brightness"] * 0.4
            warnings.append(MetricWarning(
                subsystem="image",
                metric="brightness",
                severity="caution",
                measured_value=metrics.brightness,
                expected_range=f"{bright_p5:.0f} - {bright_p95:.0f}",
                explanation="Brightness deviates moderately from calibrated baseline conditions.",
                recommendation="Verify ambient illumination matches normal operating baseline."
            ))

        # 3. Contrast Evaluation
        contrast_base = baseline_stats.get("contrast", {"median": 55.0, "mad": 15.0, "p5": 25.0, "p95": 85.0})
        contrast_median = contrast_base.get("median", 55.0)
        contrast_mad = contrast_base.get("mad", 15.0)
        contrast_p5 = contrast_base.get("p5", 25.0)
        contrast_dev = (metrics.contrast - contrast_median) / (contrast_mad if contrast_mad > 0 else 1.0)
        deviations["contrast"] = float(round(contrast_dev, 2))

        if metrics.contrast < 18.0:
            penalties += cls.IMAGE_WEIGHTS["contrast"] * 1.0
            warnings.append(MetricWarning(
                subsystem="image",
                metric="contrast",
                severity="warning",
                measured_value=metrics.contrast,
                expected_range=f"≥ {contrast_p5:.1f}",
                explanation="Extremely low dynamic contrast detected. Flat tonal range makes differentiating foreground objects difficult.",
                recommendation="Avoid foggy conditions, backlighting, or flat uniform lighting."
            ))
        elif metrics.contrast < contrast_p5 or contrast_dev < -2.0:
            penalties += cls.IMAGE_WEIGHTS["contrast"] * 0.5
            warnings.append(MetricWarning(
                subsystem="image",
                metric="contrast",
                severity="caution",
                measured_value=metrics.contrast,
                expected_range=f"≥ {contrast_p5:.1f}",
                explanation="Contrast is noticeably below normal baseline.",
                recommendation="Improve scene lighting contrast."
            ))

        # 4. Noise Evaluation
        noise_base = baseline_stats.get("noise_estimate", {"median": 4.5, "mad": 2.0, "p5": 1.0, "p95": 12.0})
        noise_median = noise_base.get("median", 4.5)
        noise_mad = noise_base.get("mad", 2.0)
        noise_p95 = noise_base.get("p95", 12.0)
        noise_dev = (metrics.noise_estimate - noise_median) / (noise_mad if noise_mad > 0 else 1.0)
        deviations["noise_estimate"] = float(round(noise_dev, 2))

        if metrics.noise_estimate > 20.0:
            penalties += cls.IMAGE_WEIGHTS["noise"] * 1.0
            warnings.append(MetricWarning(
                subsystem="image",
                metric="noise_estimate",
                severity="warning",
                measured_value=metrics.noise_estimate,
                expected_range=f"≤ {noise_p95:.1f}",
                explanation="High high-frequency noise or sensor grain detected. Noise artifacts may cause false feature activations.",
                recommendation="Lower camera ISO gain or provide stronger scene illumination."
            ))
        elif metrics.noise_estimate > noise_p95 or noise_dev > 2.5:
            penalties += cls.IMAGE_WEIGHTS["noise"] * 0.5
            warnings.append(MetricWarning(
                subsystem="image",
                metric="noise_estimate",
                severity="caution",
                measured_value=metrics.noise_estimate,
                expected_range=f"≤ {noise_p95:.1f}",
                explanation="Noise level exceeds typical baseline distribution.",
                recommendation="Check camera sensor settings and reduce gain if possible."
            ))

        # 5. Colour Cast Evaluation
        cast_base = baseline_stats.get("color_cast_delta", {"median": 12.0, "mad": 8.0, "p5": 2.0, "p95": 32.0})
        cast_median = cast_base.get("median", 12.0)
        cast_mad = cast_base.get("mad", 8.0)
        cast_p95 = cast_base.get("p95", 32.0)
        cast_dev = (metrics.color_cast_delta - cast_median) / (cast_mad if cast_mad > 0 else 1.0)
        deviations["color_cast_delta"] = float(round(cast_dev, 2))

        if metrics.color_cast_delta > 55.0:
            penalties += cls.IMAGE_WEIGHTS["color_cast"] * 1.0
            warnings.append(MetricWarning(
                subsystem="image",
                metric="color_cast",
                severity="warning",
                measured_value=metrics.color_cast_delta,
                expected_range=f"≤ {cast_p95:.1f}",
                explanation="Severe unnatural colour tint detected across RGB channels. Downstream classifiers reliant on color features may misclassify.",
                recommendation="Perform white-balance calibration on the camera."
            ))
        elif metrics.color_cast_delta > cast_p95 or cast_dev > 2.5:
            penalties += cls.IMAGE_WEIGHTS["color_cast"] * 0.4
            warnings.append(MetricWarning(
                subsystem="image",
                metric="color_cast",
                severity="caution",
                measured_value=metrics.color_cast_delta,
                expected_range=f"≤ {cast_p95:.1f}",
                explanation="Noticeable colour imbalance relative to baseline.",
                recommendation="Review ambient lighting color temperature."
            ))

        # 6. Model Prediction Entropy & Margin
        if predictions:
            # High entropy (> 4.5 for ImageNet) indicates spread out probability mass across classes
            if predictions.entropy > 4.5 or predictions.margin < 0.05:
                penalties += cls.IMAGE_WEIGHTS["model_entropy"] * 0.8
                warnings.append(MetricWarning(
                    subsystem="image",
                    metric="model_uncertainty",
                    severity="caution",
                    measured_value=predictions.entropy,
                    expected_range="entropy ≤ 3.5",
                    explanation=f"High classification entropy ({predictions.entropy:.2f}) and narrow top-2 margin ({predictions.margin:.2f}) indicate model indecision.",
                    recommendation="The model cannot confidently differentiate top classes. Provide clearer imagery or corroborating inputs."
                ))

        reliability = max(0.0, min(100.0, 100.0 - penalties))
        risk = max(0.0, min(100.0, 100.0 - reliability))

        if any(w.severity == "warning" for w in warnings) or reliability < 60.0:
            status = "warning"
        elif any(w.severity == "caution" for w in warnings) or reliability < 80.0:
            status = "caution"
        else:
            status = "normal"

        return float(round(reliability, 1)), float(round(risk, 1)), status, warnings, deviations

    @classmethod
    def evaluate_audio(cls,
                       metrics: AudioQualityMetrics,
                       classification: AudioClassificationResult,
                       baseline_stats: Dict[str, Any]) -> Tuple[float, float, str, List[MetricWarning], Dict[str, float]]:
        """
        Evaluates acoustic quality and speech suitability.
        Returns: (reliability_score, risk_score, status, warnings, deviations)
        """
        penalties = 0.0
        warnings: List[MetricWarning] = []
        deviations: Dict[str, float] = {}

        # 1. Clipping / Saturation
        clip_base = baseline_stats.get("clipping_pct", {"median": 0.0, "mad": 0.1, "p95": 0.5})
        clip_p95 = clip_base.get("p95", 0.5)
        deviations["clipping_pct"] = float(round(metrics.clipping_pct, 2))

        if metrics.clipping_pct > 2.0:
            penalties += cls.AUDIO_WEIGHTS["clipping"] * 1.0
            warnings.append(MetricWarning(
                subsystem="audio",
                metric="clipping",
                severity="warning",
                measured_value=metrics.clipping_pct,
                expected_range=f"≤ {clip_p95:.1f}%",
                explanation=f"Severe audio clipping detected ({metrics.clipping_pct:.1f}% samples clipped). Non-linear waveform squaring corrupts speech harmonics.",
                recommendation="Lower the microphone input gain or move further away from the mic."
            ))
        elif metrics.clipping_pct > clip_p95:
            penalties += cls.AUDIO_WEIGHTS["clipping"] * 0.4
            warnings.append(MetricWarning(
                subsystem="audio",
                metric="clipping",
                severity="caution",
                measured_value=metrics.clipping_pct,
                expected_range=f"≤ {clip_p95:.1f}%",
                explanation="Minor audio signal peaks are exceeding maximum digital dynamic range.",
                recommendation="Slightly lower microphone recording level."
            ))

        # 2. Loudness / RMS in dBFS
        rms_base = baseline_stats.get("rms_dbfs", {"median": -22.0, "mad": 5.0, "p5": -34.0, "p95": -12.0})
        rms_median = rms_base.get("median", -22.0)
        rms_mad = rms_base.get("mad", 5.0)
        rms_p5 = rms_base.get("p5", -34.0)
        rms_p95 = rms_base.get("p95", -12.0)
        rms_dev = (metrics.rms_dbfs - rms_median) / (rms_mad if rms_mad > 0 else 1.0)
        deviations["rms_dbfs"] = float(round(rms_dev, 2))

        if metrics.rms_dbfs < -42.0:
            penalties += cls.AUDIO_WEIGHTS["loudness"] * 1.0
            warnings.append(MetricWarning(
                subsystem="audio",
                metric="loudness",
                severity="warning",
                measured_value=metrics.rms_dbfs,
                expected_range=f"{rms_p5:.1f} to {rms_p95:.1f} dBFS",
                explanation="Audio signal is extremely faint or near-silent. Speech recognition engines will fail to transcribe quiet phonemes.",
                recommendation="Speak closer to the microphone or increase input gain."
            ))
        elif metrics.rms_dbfs < rms_p5:
            penalties += cls.AUDIO_WEIGHTS["loudness"] * 0.5
            warnings.append(MetricWarning(
                subsystem="audio",
                metric="loudness",
                severity="caution",
                measured_value=metrics.rms_dbfs,
                expected_range=f"{rms_p5:.1f} to {rms_p95:.1f} dBFS",
                explanation="Audio level is quieter than calibrated baseline.",
                recommendation="Speak clearly towards the microphone."
            ))

        # 3. Estimated SNR (Signal to Noise Ratio)
        snr_base = baseline_stats.get("snr_db", {"median": 24.0, "mad": 5.0, "p5": 12.0, "p95": 38.0})
        snr_median = snr_base.get("median", 24.0)
        snr_mad = snr_base.get("mad", 5.0)
        snr_p5 = snr_base.get("p5", 12.0)
        snr_dev = (metrics.snr_db - snr_median) / (snr_mad if snr_mad > 0 else 1.0)
        deviations["snr_db"] = float(round(snr_dev, 2))

        if metrics.snr_db < 8.0:
            penalties += cls.AUDIO_WEIGHTS["snr"] * 1.0
            warnings.append(MetricWarning(
                subsystem="audio",
                metric="snr",
                severity="warning",
                measured_value=metrics.snr_db,
                expected_range=f"≥ {snr_p5:.1f} dB",
                explanation=f"Very low Signal-to-Noise Ratio ({metrics.snr_db:.1f} dB). Ambient acoustic noise severely masks speech.",
                recommendation="Move to a quieter room, use a directional headset mic, or eliminate background noise sources."
            ))
        elif metrics.snr_db < snr_p5:
            penalties += cls.AUDIO_WEIGHTS["snr"] * 0.5
            warnings.append(MetricWarning(
                subsystem="audio",
                metric="snr",
                severity="caution",
                measured_value=metrics.snr_db,
                expected_range=f"≥ {snr_p5:.1f} dB",
                explanation="Signal-to-noise ratio is lower than optimal baseline.",
                recommendation="Minimize ambient background noise."
            ))

        # 4. Background Noise Level
        noise_base = baseline_stats.get("background_noise_db", {"median": -45.0, "mad": 6.0, "p95": -32.0})
        noise_p95 = noise_base.get("p95", -32.0)
        deviations["background_noise_db"] = float(round(metrics.background_noise_db, 2))

        if metrics.background_noise_db > -28.0:
            penalties += cls.AUDIO_WEIGHTS["noise_floor"] * 0.8
            warnings.append(MetricWarning(
                subsystem="audio",
                metric="background_noise",
                severity="warning",
                measured_value=metrics.background_noise_db,
                expected_range=f"≤ {noise_p95:.1f} dBFS",
                explanation="Unusually loud ambient background noise floor detected.",
                recommendation="Close windows, turn off loud fans or HVAC, or use a noise-canceling microphone."
            ))

        # 5. CNN Acoustic Condition Check
        if classification:
            primary = classification.primary_condition
            if primary in ["Wind / Low-Frequency Noise", "Mechanical Hum", "Clipped / Distorted"]:
                penalties += cls.AUDIO_WEIGHTS["condition"] * 0.9
                warnings.append(MetricWarning(
                    subsystem="audio",
                    metric="acoustic_condition",
                    severity="warning",
                    measured_value=classification.condition_probabilities.get(primary, 0.0),
                    expected_range="Clean Speech",
                    explanation=f"Acoustic classifier detected '{primary}' as the dominant acoustic condition.",
                    recommendation="Eliminate the interfering acoustic source or reposition microphone."
                ))
            elif primary == "Background Noise":
                penalties += cls.AUDIO_WEIGHTS["condition"] * 0.5
                warnings.append(MetricWarning(
                    subsystem="audio",
                    metric="acoustic_condition",
                    severity="caution",
                    measured_value=classification.condition_probabilities.get(primary, 0.0),
                    expected_range="Clean Speech",
                    explanation="Acoustic classifier detected significant environmental noise.",
                    recommendation="Ensure the speaker is positioned directly in front of the mic."
                ))

        reliability = max(0.0, min(100.0, 100.0 - penalties))
        risk = max(0.0, min(100.0, 100.0 - reliability))

        if any(w.severity == "warning" for w in warnings) or reliability < 60.0:
            status = "warning"
        elif any(w.severity == "caution" for w in warnings) or reliability < 80.0:
            status = "caution"
        else:
            status = "normal"

        return float(round(reliability, 1)), float(round(risk, 1)), status, warnings, deviations
