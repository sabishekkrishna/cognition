import cv2
import numpy as np
from typing import Tuple, Optional
from ..schemas.schemas import ImageDegradationConfig

def apply_image_degradation(image_bgr: np.ndarray, config: ImageDegradationConfig) -> Tuple[np.ndarray, str]:
    """
    Applies synthetic physical and sensor degradation to an image.
    Returns: (degraded_image_bgr, degradation_description)
    """
    img = image_bgr.copy().astype(np.float32)
    applied_effects = []

    # 1. Darkness or Excessive Brightness
    if config.brightness_factor != 1.0:
        img = img * config.brightness_factor
        img = np.clip(img, 0, 255)
        if config.brightness_factor < 1.0:
            applied_effects.append(f"Darkness (factor {config.brightness_factor:.2f})")
        else:
            applied_effects.append(f"Excessive Brightness (factor {config.brightness_factor:.2f})")

    # 2. Reduced Contrast
    if config.contrast_factor != 1.0:
        mean_intensity = np.mean(img)
        img = mean_intensity + config.contrast_factor * (img - mean_intensity)
        img = np.clip(img, 0, 255)
        applied_effects.append(f"Reduced Contrast (factor {config.contrast_factor:.2f})")

    # 3. Gaussian Blur
    if config.gaussian_blur_kernel > 1:
        # Kernel size must be odd
        ksize = int(config.gaussian_blur_kernel)
        if ksize % 2 == 0:
            ksize += 1
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        applied_effects.append(f"Gaussian Blur (kernel {ksize}x{ksize})")

    # 4. Colour Tint
    if config.color_tint:
        tint = config.color_tint.lower()
        intensity = np.clip(config.color_tint_intensity, 0.0, 1.0)
        # BGR channels: [0]=Blue, [1]=Green, [2]=Red
        if tint == "red":
            img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + intensity * 1.5), 0, 255)
            img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - intensity * 0.5), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 - intensity * 0.5), 0, 255)
            applied_effects.append(f"Red Colour Tint ({int(intensity*100)}%)")
        elif tint == "green":
            img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 + intensity * 1.5), 0, 255)
            img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - intensity * 0.5), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 - intensity * 0.5), 0, 255)
            applied_effects.append(f"Green Colour Tint ({int(intensity*100)}%)")
        elif tint == "blue":
            img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 + intensity * 1.5), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 - intensity * 0.5), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 - intensity * 0.5), 0, 255)
            applied_effects.append(f"Blue Colour Tint ({int(intensity*100)}%)")
        elif tint == "yellow":
            img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + intensity * 1.2), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 + intensity * 1.2), 0, 255)
            img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - intensity * 0.8), 0, 255)
            applied_effects.append(f"Yellow/Warm Cast ({int(intensity*100)}%)")

    # 5. Gaussian Noise
    if config.gaussian_noise_sigma > 0:
        noise = np.random.normal(0, config.gaussian_noise_sigma, img.shape).astype(np.float32)
        img = img + noise
        img = np.clip(img, 0, 255)
        applied_effects.append(f"Gaussian Noise (sigma {config.gaussian_noise_sigma:.1f})")

    degraded_uint8 = img.astype(np.uint8)
    summary = ", ".join(applied_effects) if applied_effects else "None (Clean Original)"
    return degraded_uint8, summary
