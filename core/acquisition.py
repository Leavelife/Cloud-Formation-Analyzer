"""Image acquisition and color-channel conversion helpers."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFile


def rgb_from_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def bgr_from_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to OpenCV BGR."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image.")
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def load_rgb_image(file_obj) -> tuple[np.ndarray, str]:
    """Load an uploaded image as RGB uint8 with PIL."""
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        image = Image.open(file_obj)
        image.load()
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Could not read image: {exc}") from exc
    if rgb.size == 0 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("Uploaded image has no pixels.")
    return rgb, str(getattr(image, "format", "") or "Unknown")
