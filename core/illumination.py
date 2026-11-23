"""Image-derived illumination diagnostics and user-defined sunlight regions."""

from __future__ import annotations

import cv2
import numpy as np


def illumination_analysis(
    image_rgb: np.ndarray,
    sunlight_region: tuple[int, int, int, int] | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return brightness, warmth, local illumination, and a user region mask.

    Warm-bright pixels are diagnostic cues, not solar localization. A sun
    interference region is only marked when the user explicitly supplies one.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")
    rgb = image_rgb.astype(np.float32)
    if image_rgb.dtype == np.uint8:
        rgb /= 255.0
    red, green, blue = (rgb[:, :, i] for i in range(3))
    brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    warmth = np.clip(red - blue, 0.0, 1.0)
    local_illumination = cv2.GaussianBlur(brightness, (0, 0), sigmaX=9, sigmaY=9)
    if valid_mask is None:
        valid_mask = np.ones(brightness.shape, dtype=bool)
    elif valid_mask.shape != brightness.shape:
        raise ValueError("Valid-sky mask dimensions must match the image.")
    valid_brightness = brightness[valid_mask]
    if valid_brightness.size == 0:
        raise ValueError("No valid pixels available for illumination analysis.")
    bright_cutoff = np.quantile(valid_brightness, 0.75)
    warm_bright = (brightness >= bright_cutoff) & (warmth >= 0.08) & valid_mask

    user_region = np.zeros(brightness.shape, dtype=bool)
    if sunlight_region is not None:
        x1, y1, x2, y2 = map(int, sunlight_region)
        h, w = brightness.shape
        x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
        y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
        if x1 < x2 and y1 < y2:
            user_region[y1:y2, x1:x2] = True
    return {
        "brightness": brightness,
        "warmth": warmth,
        "local_illumination": local_illumination,
        "warm_bright_candidates": warm_bright,
        "user_sunlight_region": user_region,
    }
