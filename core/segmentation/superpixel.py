"""Superpixel label generation and legacy local-feature segmentation."""

from __future__ import annotations

import cv2
import numpy as np


def _watershed_regions(
    image_rgb: np.ndarray,
    number_of_regions: int,
    compactness: float,
) -> np.ndarray:
    """Produce seeded watershed regions when explicitly selected."""
    height, width = image_rgb.shape[:2]
    spacing = max(4, int(np.sqrt((height * width) / max(number_of_regions, 1))))
    markers = np.zeros((height, width), dtype=np.int32)
    marker_id = 1
    for y in range(spacing // 2, height, spacing):
        for x in range(spacing // 2, width, spacing):
            markers[y, x] = marker_id
            marker_id += 1
    if marker_id <= 1:
        markers[height // 2, width // 2] = 1

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # Compactness controls gradient smoothing, not geometric SLIC compactness.
    sigma = max(0.0, min(5.0, float(compactness) / 20.0))
    if sigma:
        image_bgr = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    cv2.watershed(image_bgr, markers)
    return markers


def create_superpixel_labels(
    image_rgb: np.ndarray,
    number_of_superpixels: int = 400,
    compactness: float = 10.0,
    segmentation_method: str = "SLIC",
) -> tuple[np.ndarray, str]:
    """Create SLIC or explicitly requested watershed region labels."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image.")
    if number_of_superpixels < 1 or compactness <= 0:
        raise ValueError("Superpixel count and compactness must be positive.")
    if segmentation_method not in ("SLIC", "Watershed"):
        raise ValueError(f"Unknown superpixel method: {segmentation_method}")

    height, width = image_rgb.shape[:2]
    if segmentation_method == "SLIC":
        if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "createSuperpixelSLIC"):
            raise RuntimeError(
                "SLIC requires opencv-contrib-python-headless; install the project requirements."
            )
        region_size = max(2, int(np.sqrt(height * width / number_of_superpixels)))
        slic = cv2.ximgproc.createSuperpixelSLIC(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB),
            algorithm=cv2.ximgproc.SLIC,
            region_size=region_size,
            ruler=float(compactness),
        )
        slic.iterate(10)
        return slic.getLabels(), "SLIC"

    return _watershed_regions(image_rgb, number_of_superpixels, compactness), "OpenCV watershed"


def segment_superpixels(
    image_rgb: np.ndarray,
    feature: np.ndarray,
    valid_mask: np.ndarray | None = None,
    number_of_superpixels: int = 400,
    compactness: float = 10.0,
    local_threshold_scale: float = 0.25,
    local_feature_ceiling: float = -0.65,
    segmentation_method: str = "SLIC",
    labels: np.ndarray | None = None,
    implementation: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Locally threshold a cloud-low feature by coherent image regions.

    SLIC is the default. Watershed can be explicitly selected as an alternative
    and is identified as such in diagnostics.
    Cloud candidates lie below each region's median minus a local dispersion
    term. This preserves thin regions better than large-kernel morphology.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image.")
    if feature.shape != image_rgb.shape[:2]:
        raise ValueError("Feature map must match the RGB image dimensions.")
    if valid_mask is None:
        valid_mask = np.ones(feature.shape, dtype=bool)
    elif valid_mask.shape != feature.shape:
        raise ValueError("Valid-sky mask must match the feature-map dimensions.")
    if local_threshold_scale < 0 or not np.isfinite(local_feature_ceiling):
        raise ValueError("Local threshold scale must be non-negative and ceiling finite.")
    if not np.isfinite(feature).all():
        raise ValueError("Feature map must contain only finite values.")
    height, width = feature.shape

    if labels is None:
        labels, implementation = create_superpixel_labels(
            image_rgb, number_of_superpixels, compactness, segmentation_method
        )
    elif labels.shape != feature.shape:
        raise ValueError("Superpixel labels must match the feature-map dimensions.")
    else:
        implementation = implementation or segmentation_method

    result = np.zeros((height, width), dtype=np.uint8)
    local_thresholds = []
    for region_id in np.unique(labels):
        if region_id < 0:
            continue
        region = (labels == region_id) & valid_mask
        samples = feature[region]
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            continue
        median = float(np.median(samples))
        spread = float(np.median(np.abs(samples - median)) * 1.4826)
        threshold = min(
            median - float(local_threshold_scale) * spread,
            float(local_feature_ceiling),
        )
        result[region] = np.where(feature[region] <= threshold, 255, 0).astype(np.uint8)
        local_thresholds.append(threshold)

    boundaries = np.zeros((height, width), dtype=np.uint8)
    boundaries[1:, :] |= (labels[1:, :] != labels[:-1, :]).astype(np.uint8) * 255
    boundaries[:, 1:] |= (labels[:, 1:] != labels[:, :-1]).astype(np.uint8) * 255
    return result, boundaries, {
        "implementation": implementation,
        "region_count": int(np.unique(labels[labels >= 0]).size),
        "mean_local_threshold": float(np.mean(local_thresholds)) if local_thresholds else 0.0,
    }
