"""Optional local texture evidence."""

from __future__ import annotations

import cv2
import numpy as np


def extract_texture(
    image_rgb: np.ndarray,
    window_size: int = 9,
    entropy_bins: int = 16,
    adaptive_threshold_scale: float = 0.25,
) -> dict[str, np.ndarray]:
    """Compute local texture, signed contrast, and an adaptive gray threshold."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")
    size = int(window_size)
    if size < 3:
        raise ValueError("Texture window must be at least 3.")
    if size % 2 == 0:
        size += 1
    if not 2 <= entropy_bins <= 32:
        raise ValueError("entropy_bins must be between 2 and 32.")
    if adaptive_threshold_scale < 0 or not np.isfinite(adaptive_threshold_scale):
        raise ValueError("Adaptive threshold scale must be finite and non-negative.")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = cv2.boxFilter(gray, -1, (size, size), normalize=True)
    mean_square = cv2.boxFilter(gray * gray, -1, (size, size), normalize=True)
    local_std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    local_contrast = gray - mean
    contrast_z = local_contrast / (local_std + 1e-3)

    quantized = np.minimum((gray * entropy_bins).astype(np.int32), entropy_bins - 1)
    entropy = np.zeros(gray.shape, dtype=np.float32)
    for level in range(entropy_bins):
        indicator = (quantized == level).astype(np.float32)
        probability = cv2.boxFilter(indicator, -1, (size, size), normalize=True)
        entropy -= np.where(probability > 0, probability * np.log2(np.maximum(probability, 1e-12)), 0.0)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    return {
        "local_std": local_std,
        "local_entropy": entropy,
        "gradient": gradient,
        "local_contrast": local_contrast,
        "contrast_z": contrast_z,
        "adaptive_threshold": mean + float(adaptive_threshold_scale) * local_std,
        "adaptive_bright": gray > (
            mean + float(adaptive_threshold_scale) * local_std + 1e-6
        ),
    }
