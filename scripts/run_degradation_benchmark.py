import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
import scipy.io.wavfile as wavfile
from backend.app.image.metrics import analyze_image_quality
from backend.app.image.degradation import apply_image_degradation
from backend.app.schemas.schemas import ImageDegradationConfig, AudioDegradationConfig
from backend.app.audio.metrics import analyze_audio_quality
from backend.app.audio.degradation import apply_audio_degradation, samples_to_wav_bytes
from backend.app.baselines.manager import BaselineManager
from backend.app.monitoring.scoring import ReliabilityScoringEngine
from backend.app.image.classifier import get_classifier

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "test_assets"))

def generate_assets():
    img_dir = os.path.join(ASSETS_DIR, "images")
    aud_dir = os.path.join(ASSETS_DIR, "audio")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)

    # 1. Clean image
    img = np.full((320, 320, 3), 160, dtype=np.uint8)
    cv2.circle(img, (160, 160), 90, (40, 180, 80), -1)
    cv2.circle(img, (160, 160), 40, (230, 230, 230), -1)
    cv2.rectangle(img, (50, 50), (120, 120), (220, 80, 40), -1)
    cv2.putText(img, "NORMAL-01", (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    clean_path = os.path.join(img_dir, "clean_component.jpg")
    cv2.imwrite(clean_path, img)

    # 2. Degraded: Blur
    blur_img, _ = apply_image_degradation(img, ImageDegradationConfig(gaussian_blur_kernel=21))
    cv2.imwrite(os.path.join(img_dir, "blurred_component.jpg"), blur_img)

    # 3. Degraded: Dark
    dark_img, _ = apply_image_degradation(img, ImageDegradationConfig(brightness_factor=0.15))
    cv2.imwrite(os.path.join(img_dir, "dark_component.jpg"), dark_img)

    # 4. Degraded: Noisy
    noisy_img, _ = apply_image_degradation(img, ImageDegradationConfig(gaussian_noise_sigma=35.0))
    cv2.imwrite(os.path.join(img_dir, "noisy_component.jpg"), noisy_img)

    # 5. Clean Audio
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Speech tone with speech burst and pauses
    audio = np.zeros_like(t, dtype=np.float32)
    burst_len = int(len(t) * 0.7)
    audio[:burst_len] = (0.5 * np.sin(2 * np.pi * 260 * t[:burst_len]) + 
                         0.25 * np.sin(2 * np.pi * 520 * t[:burst_len])).astype(np.float32)
    clean_aud_path = os.path.join(aud_dir, "clean_speech_harmonic.wav")
    wavfile.write(clean_aud_path, sr, (audio * 32767).astype(np.int16))

    # 6. Clipped Audio
    clipped_aud, _ = apply_audio_degradation(audio, sr, AudioDegradationConfig(clipping_gain=5.0))
    wavfile.write(os.path.join(aud_dir, "clipped_speech.wav"), sr, (clipped_aud * 32767).astype(np.int16))

    # 7. Noisy Audio
    noisy_aud, _ = apply_audio_degradation(audio, sr, AudioDegradationConfig(noise_type="white", noise_level=0.8))
    wavfile.write(os.path.join(aud_dir, "noisy_speech.wav"), sr, (noisy_aud * 32767).astype(np.int16))

    print(f"Generated test assets successfully in {ASSETS_DIR}")

def run_benchmark():
    print("=" * 60)
    print("RUNNING ON-DEVICE PERCEPTUAL RELIABILITY BENCHMARK")
    print("=" * 60)
    generate_assets()

    baseline = BaselineManager.get_or_create_default_image_baseline()
    classifier = get_classifier()

    img_dir = os.path.join(ASSETS_DIR, "images")
    test_files = [
        ("Clean Component", "clean_component.jpg"),
        ("Gaussian Blur", "blurred_component.jpg"),
        ("Severe Underexposure", "dark_component.jpg"),
        ("Sensor Noise", "noisy_component.jpg"),
    ]

    print("\nIMAGE DEGRADATION DETECTION BENCHMARK:")
    print("-" * 60)
    for label, filename in test_files:
        path = os.path.join(img_dir, filename)
        img = cv2.imread(path)
        metrics = analyze_image_quality(img)
        preds = classifier.predict(img, top_k=3)
        rel_score, risk_score, status, warnings, _ = ReliabilityScoringEngine.evaluate_image(
            metrics, preds, baseline["stats"]
        )
        print(f"[{label:22}] Reliability: {rel_score:5.1f} | Risk: {risk_score:5.1f} | Status: {status.upper():7} | Warnings: {len(warnings)}")
        for w in warnings:
            print(f"   --> {w.metric.upper()}: {w.explanation[:70]}...")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_benchmark()
