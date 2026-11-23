"""Configurable binary morphology operations."""

from __future__ import annotations

import cv2
import numpy as np


def apply_morphology(
    binary_mask: np.ndarray,
    operation: str = "Opening",
    kernel_size: int = 3,
    erosion_size: int = 0,
    dilation_size: int = 0,
) -> np.ndarray:
    """Apply erosion, dilation, opening, and/or closing with a configurable ellipse kernel."""
    mask = np.where(binary_mask > 0, 255, 0).astype(np.uint8)
    size = int(kernel_size)
    if size < 1:
        raise ValueError("Morphology kernel size must be at least 1.")
    erosion_iterations = int(erosion_size)
    dilation_iterations = int(dilation_size)
    if erosion_iterations < 0 or dilation_iterations < 0:
        raise ValueError("Morphology iteration counts cannot be negative.")
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    result = mask.copy()
    if operation == "Erosion":
        result = cv2.erode(result, kernel, iterations=max(1, erosion_iterations))
    elif operation == "Dilation":
        result = cv2.dilate(result, kernel, iterations=max(1, dilation_iterations))
    elif operation not in ("None", "Opening", "Closing", "Opening + Closing"):
        raise ValueError(f"Unknown morphology operation: {operation}")
    else:
        for _ in range(erosion_iterations):
            result = cv2.erode(result, kernel)
        for _ in range(dilation_iterations):
            result = cv2.dilate(result, kernel)
        if operation == "Opening":
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif operation == "Closing":
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        elif operation == "Opening + Closing":
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result
