"""Classical cloud segmentation methods."""

from core.segmentation.baselines import (
    hsv_segmentation,
    lab_segmentation,
    otsu_brightness_segmentation,
    rgb_to_hsv,
)

__all__ = [
    "hsv_segmentation",
    "lab_segmentation",
    "otsu_brightness_segmentation",
    "rgb_to_hsv",
]
