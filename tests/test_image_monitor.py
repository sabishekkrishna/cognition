import pytest
import numpy as np
import cv2
from backend.app.image.metrics import analyze_image_quality, compute_immerkaer_noise
from backend.app.image.classifier import get_classifier
from backend.app.image.degradation import apply_image_degradation
from backend.app.schemas.schemas import ImageDegradationConfig

def create_synthetic_test_image(pattern="normal") -> np.ndarray:
    """Creates synthetic reproducible images for test assertions."""
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    if pattern == "normal":
        cv2.circle(img, (112, 112), 60, (50, 180, 50), -1)
        cv2.rectangle(img, (30, 30), (90, 90), (200, 80, 60), -1)
        cv2.putText(img, "TEST", (35, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img

def test_opencv_metrics_calculation():
    img = create_synthetic_test_image("normal")
    metrics = analyze_image_quality(img)

    assert 0 <= metrics.brightness <= 255
    assert metrics.sharpness > 0
    assert metrics.contrast > 0
    assert metrics.width == 224
    assert metrics.height == 224
    assert metrics.resolution_mp > 0
    assert 0 <= metrics.underexposure_pct <= 100
    assert 0 <= metrics.overexposure_pct <= 100

def test_mobilenetv3_inference_and_embeddings():
    classifier = get_classifier()
    img = create_synthetic_test_image("normal")
    preds = classifier.predict(img, top_k=5)

    assert len(preds.top_predictions) == 5
    assert 0.0 <= preds.top_probability <= 1.0
    assert preds.entropy >= 0.0
    assert preds.margin >= 0.0
    assert preds.embedding_dimension == 576
    assert preds.embedding_sample is not None
    assert len(preds.embedding_sample) > 0

def test_blur_degradation_detection():
    img = create_synthetic_test_image("normal")
    clean_metrics = analyze_image_quality(img)

    # Apply heavy Gaussian blur
    config = ImageDegradationConfig(gaussian_blur_kernel=25)
    blurred_img, desc = apply_image_degradation(img, config)
    blurred_metrics = analyze_image_quality(blurred_img)

    # Blurring must measurably drop Laplacian variance sharpness
    assert blurred_metrics.sharpness < clean_metrics.sharpness
    assert "Gaussian Blur" in desc

def test_darkness_and_brightness_degradation():
    img = create_synthetic_test_image("normal")
    clean_metrics = analyze_image_quality(img)

    # Darken heavily to trigger underexposure
    dark_config = ImageDegradationConfig(brightness_factor=0.05)
    dark_img, _ = apply_image_degradation(img, dark_config)
    dark_metrics = analyze_image_quality(dark_img)
    assert dark_metrics.brightness < clean_metrics.brightness
    assert dark_metrics.underexposure_pct > clean_metrics.underexposure_pct

    # Brighten
    bright_config = ImageDegradationConfig(brightness_factor=2.0)
    bright_img, _ = apply_image_degradation(img, bright_config)
    bright_metrics = analyze_image_quality(bright_img)
    assert bright_metrics.brightness > clean_metrics.brightness

def test_noise_degradation_detection():
    img = create_synthetic_test_image("normal")
    clean_metrics = analyze_image_quality(img)

    noise_config = ImageDegradationConfig(gaussian_noise_sigma=35.0)
    noisy_img, _ = apply_image_degradation(img, noise_config)
    noisy_metrics = analyze_image_quality(noisy_img)

    # Immerkaer noise estimator must detect noise increase
    assert noisy_metrics.noise_estimate > clean_metrics.noise_estimate
