import pytest
import numpy as np
from backend.app.baselines.manager import compute_robust_distribution, BaselineManager
from backend.app.monitoring.scoring import ReliabilityScoringEngine
from backend.app.schemas.schemas import ImageQualityMetrics, ModelPredictions, PredictionItem

def test_robust_distribution_stats():
    # Outlier-contaminated distribution
    values = [10.0, 11.0, 10.5, 12.0, 11.5, 9.8, 100.0] # 100 is an outlier
    stats = compute_robust_distribution(values)

    assert stats["median"] < 15.0 # median resistant to outlier
    assert stats["mad"] < 10.0    # MAD resistant to outlier
    assert stats["min"] == 9.8
    assert stats["max"] == 100.0

def test_image_scoring_and_risk_inversion():
    baseline = BaselineManager.get_or_create_default_image_baseline()
    baseline_stats = baseline["stats"]

    # Normal clean image metrics
    clean_metrics = ImageQualityMetrics(
        brightness=128.0,
        sharpness=220.0,
        contrast=60.0,
        color_mean_r=125.0,
        color_mean_g=127.0,
        color_mean_b=126.0,
        color_cast_delta=2.0,
        noise_estimate=4.0,
        underexposure_pct=0.5,
        overexposure_pct=0.5,
        width=224,
        height=224,
        resolution_mp=0.05
    )

    clean_preds = ModelPredictions(
        model_name="test",
        top_predictions=[PredictionItem(class_id=1, class_name="cat", probability=0.85)],
        top_class="cat",
        top_probability=0.85,
        entropy=1.2,
        margin=0.7,
        embedding_dimension=576
    )

    rel_score, risk_score, status, warnings, _ = ReliabilityScoringEngine.evaluate_image(
        clean_metrics, clean_preds, baseline_stats
    )

    assert rel_score >= 85.0
    assert risk_score <= 15.0
    assert rel_score + risk_score == pytest.approx(100.0, abs=0.1)
    assert status == "normal"
    assert len(warnings) == 0

    # Heavily blurred image metrics
    blurred_metrics = ImageQualityMetrics(
        brightness=128.0,
        sharpness=15.0, # Critically blurred
        contrast=30.0,
        color_mean_r=125.0,
        color_mean_g=127.0,
        color_mean_b=126.0,
        color_cast_delta=2.0,
        noise_estimate=4.0,
        underexposure_pct=0.5,
        overexposure_pct=0.5,
        width=224,
        height=224,
        resolution_mp=0.05
    )

    b_rel, b_risk, b_status, b_warnings, _ = ReliabilityScoringEngine.evaluate_image(
        blurred_metrics, clean_preds, baseline_stats
    )

    assert b_rel < 80.0
    assert b_risk > 20.0
    assert b_status in ["caution", "warning"]
    assert any(w.metric == "sharpness" for w in b_warnings)
