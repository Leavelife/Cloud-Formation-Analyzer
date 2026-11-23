"""Normalized Saturation/Value contrast feature (NSV)."""

from __future__ import annotations

import cv2
import numpy as np


def extract_nsv(image_rgb: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Compute ``(S - V) / (S + V + epsilon)`` from normalized HSV channels.

    OpenCV HSV S and V bytes are converted to [0, 1] before the normalized
    difference. The epsilon makes black pixels finite; output is bounded
    approximately to [-1, 1].
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and greater than zero.")
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    return (saturation - value) / (saturation + value + epsilon)
