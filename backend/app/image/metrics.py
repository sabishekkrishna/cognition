import cv2
import numpy as np
import math
from typing import Dict, Any, Tuple
from ..schemas.schemas import ImageQualityMetrics

def compute_immerkaer_noise(gray: np.ndarray) -> float:
    """
    Immerkaer fast noise estimator:
    Estimates standard deviation of additive zero-mean Gaussian noise from a single image
    using a 3x3 Laplacian mask:
    [ 1 -2  1]
    [-2  4 -2]
    [ 1 -2  1]
    Formula: sigma = sqrt(pi / 2) * (1 / (6 * (W - 2) * (H - 2))) * sum(|conv|)
    """
    H, W = gray.shape
    if H < 10 or W < 10:
        return 0.0

    mask = np.array([[1, -2, 1],
                     [-2,  4, -2],
                     [1, -2, 1]], dtype=np.float64)

    # Convert to float64 to prevent overflow
    gray_f = gray.astype(np.float64)
    conv = cv2.filter2D(gray_f, -1, mask)

    # Trim boundary artifacts (1 pixel on each side)
    conv_interior = conv[1:-1, 1:-1]
    sum_abs = np.sum(np.abs(conv_interior))
    count = (W - 2) * (H - 2)

    sigma = math.sqrt(0.5 * math.pi) * (1.0 / (6.0 * count)) * sum_abs
    return float(round(sigma, 3))

def analyze_image_quality(image_bgr: np.ndarray) -> ImageQualityMetrics:
    """
    Calculates key OpenCV visual quality statistics for an image.
    Accepts BGR numpy array from cv2.imread or cv2.imdecode.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid or empty image array provided.")

    height, width = image_bgr.shape[:2]
    resolution_mp = round((height * width) / 1_000_000.0, 3)

    # Grayscale for luminance metrics
    if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(image_bgr)
        mean_r = float(np.mean(r))
        mean_g = float(np.mean(g))
        mean_b = float(np.mean(b))
    else:
        gray = image_bgr
        mean_r = mean_g = mean_b = float(np.mean(gray))

    # 1. Brightness: Mean grayscale luminance intensity [0-255]
    brightness = float(np.mean(gray))

    # 2. Sharpness: Variance of the Laplacian as high-frequency edge indicator
    # Higher value = sharper edges; low values (< 60) indicate blur or defocus
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # 3. Contrast: Standard deviation of pixel intensities [0-127.5]
    contrast = float(np.std(gray))

    # 4. Colour balance & cast delta: Max channel disparity
    cast_delta = float(max(abs(mean_r - mean_g), abs(mean_r - mean_b), abs(mean_g - mean_b)))

    # 5. Noise estimate via Immerkaer Laplacian estimator
    noise_estimate = compute_immerkaer_noise(gray)

    # 6. Exposure indicators: Near-black (<10) and saturated (>245) percentages
    total_pixels = float(height * width)
    underexposure_pct = float((np.count_nonzero(gray < 10) / total_pixels) * 100.0)
    overexposure_pct = float((np.count_nonzero(gray > 245) / total_pixels) * 100.0)

    return ImageQualityMetrics(
        brightness=round(brightness, 2),
        sharpness=round(sharpness, 2),
        contrast=round(contrast, 2),
        color_mean_r=round(mean_r, 2),
        color_mean_g=round(mean_g, 2),
        color_mean_b=round(mean_b, 2),
        color_cast_delta=round(cast_delta, 2),
        noise_estimate=round(noise_estimate, 2),
        underexposure_pct=round(underexposure_pct, 2),
        overexposure_pct=round(overexposure_pct, 2),
        width=int(width),
        height=int(height),
        resolution_mp=resolution_mp
    )
