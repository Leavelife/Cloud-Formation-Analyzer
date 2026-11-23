"""Region-of-interest helpers for the image-processing pipeline."""

from __future__ import annotations

import cv2
import numpy as np


def validate_roi(
    roi: tuple[int, int, int, int] | list[int] | None,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Validate and clamp an (x1, y1, x2, y2) ROI against the image."""
    if len(image_shape) < 2:
        raise ValueError("Image shape must contain height and width.")
    height, width = int(image_shape[0]), int(image_shape[1])
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    if roi is None:
        return (0, 0, width, height)
    if len(roi) != 4:
        raise ValueError("ROI must contain exactly (x1, y1, x2, y2).")

    x1, y1, x2, y2 = (int(value) for value in roi)
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty or invalid ROI: width and height must be greater than zero.")
    return (x1, y1, x2, y2)


def build_valid_sky_mask(
    image_shape: tuple[int, ...],
    roi: tuple[int, int, int, int],
    polygon_points: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build a full-frame boolean-valid mask for a rectangular or polygon ROI."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = validate_roi(roi, image_shape)
    valid = np.zeros((height, width), dtype=np.uint8)
    valid[y1:y2, x1:x2] = 1

    if polygon_points is not None and len(polygon_points) > 0:
        if len(polygon_points) < 3:
            raise ValueError("A validity polygon must contain at least three points.")
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array(
            [[max(0, min(width - 1, x)), max(0, min(height - 1, y))]
             for x, y in polygon_points],
            dtype=np.int32,
        )
        if len(points) >= 3:
            cv2.fillPoly(polygon_mask, [points.reshape((-1, 1, 2))], 1)
            valid = cv2.bitwise_and(valid, polygon_mask)
    return valid.astype(bool)
