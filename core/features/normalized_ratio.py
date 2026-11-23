"""Normalized blue/red color ratio features."""

from __future__ import annotations

import numpy as np


def extract_normalized_blue_red(
    image_rgb: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute ``(B-R)/(B+R+epsilon)`` using normalized RGB channels."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and greater than zero.")
    rgb = image_rgb.astype(np.float32)
    if image_rgb.dtype == np.uint8:
        rgb /= 255.0
    elif np.nanmin(rgb) < 0 or np.nanmax(rgb) > 1:
        raise ValueError("Floating-point RGB values must be in [0, 1].")
    red, blue = rgb[:, :, 0], rgb[:, :, 2]
    return (blue - red) / (blue + red + epsilon)
