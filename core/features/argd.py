"""Adjustable red-green difference feature for sunlight-aware cloud analysis."""

from __future__ import annotations

import numpy as np


def extract_argd(
    image_rgb: np.ndarray,
    k: float = 1.0,
    sunlight_mask: np.ndarray | None = None,
    sunlight_k: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ARGD ``k*R-G`` and its per-pixel red weight map.

    Channels are converted to floating point in [0, 1] before arithmetic.
    A user-specified sunlight mask receives ``sunlight_k``; no solar position
    or sun direction is inferred from an arbitrary photograph.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (height, width, 3).")
    if not np.isfinite(k) or (sunlight_k is not None and not np.isfinite(sunlight_k)):
        raise ValueError("ARGD weights must be finite.")

    rgb = image_rgb.astype(np.float32)
    if image_rgb.dtype == np.uint8:
        rgb /= 255.0
    elif np.nanmin(rgb) < 0 or np.nanmax(rgb) > 1:
        raise ValueError("Floating-point RGB values must be in [0, 1].")
    red, green = rgb[:, :, 0], rgb[:, :, 1]

    weights = np.full(red.shape, float(k), dtype=np.float32)
    if sunlight_mask is not None:
        mask = np.asarray(sunlight_mask, dtype=bool)
        if mask.shape != red.shape:
            raise ValueError("Sunlight mask dimensions must match the RGB image.")
        if sunlight_k is not None:
            weights[mask] = float(sunlight_k)
    return weights * red - green, weights
