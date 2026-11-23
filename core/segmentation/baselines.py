"""Baseline color-space segmentation methods."""

from __future__ import annotations

import cv2
import numpy as np


def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Convert a validated RGB uint8 image to OpenCV HSV."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB image with dtype uint8.")
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


def hsv_segmentation(image_rgb: np.ndarray, params: dict) -> np.ndarray:
    """Segment bright, low-saturation candidates with configurable HSV ranges."""
    hsv = rgb_to_hsv(image_rgb)
    lower = np.array([
        int(params.get("h_min", 0)),
        int(params.get("s_min", 0)),
        int(params.get("v_min", 150)),
    ], dtype=np.int32)
    upper = np.array([
        int(params.get("h_max", 180)),
        int(params.get("s_max", 85)),
        int(params.get("v_max", 255)),
    ], dtype=np.int32)
    if np.any(lower < 0) or np.any(upper > [180, 255, 255]) or np.any(lower > upper):
        raise ValueError("Invalid HSV threshold range.")
    return cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8))


def lab_segmentation(
    image_rgb: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Segment bright neutral chroma in CIELAB space."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    chroma = np.sqrt(
        (lab[:, :, 1].astype(np.float32) - 128.0) ** 2
        + (lab[:, :, 2].astype(np.float32) - 128.0) ** 2
    )
    l_min = int(params.get("lab_l_min", 170))
    chroma_max = float(params.get("lab_chroma_max", 28))
    if not 0 <= l_min <= 255 or chroma_max < 0:
        raise ValueError("LAB thresholds must have L in [0, 255] and a non-negative chroma limit.")
    return np.where((lightness >= l_min) & (chroma <= chroma_max), 255, 0).astype(np.uint8)


def otsu_brightness_segmentation(image_rgb: np.ndarray) -> np.ndarray:
    """Use global Otsu thresholding on grayscale brightness as a justified alternative."""
    grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _threshold, mask = cv2.threshold(
        grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask
