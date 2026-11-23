"""Analysis summaries and metric-based method selection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def coverage_summary(mask: np.ndarray, valid_mask: np.ndarray) -> dict[str, int | float]:
    """Count cloud and sky pixels using only valid ROI pixels as denominator."""
    predicted = np.asarray(mask)
    valid = np.asarray(valid_mask, dtype=bool)
    if predicted.ndim != 2 or valid.ndim != 2:
        raise ValueError("Cloud mask and valid ROI mask must be two-dimensional.")
    if predicted.shape != valid.shape:
        raise ValueError("Cloud mask and valid ROI mask dimensions must match.")
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("No valid ROI pixels are available for coverage analysis.")

    cloud = (predicted > 0) & valid
    cloud_count = int(np.count_nonzero(cloud))
    sky_count = valid_count - cloud_count
    return {
        "cloud_pixels": cloud_count,
        "sky_pixels": sky_count,
        "valid_roi_pixels": valid_count,
        "cloud_coverage": cloud_count / valid_count * 100.0,
        "sky_coverage": sky_count / valid_count * 100.0,
    }


def compare_method_coverage(
    methods: dict[str, dict[str, Any]],
    valid_mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Summarize every method that provides a same-sized output mask."""
    rows = []
    for name, method in methods.items():
        if not isinstance(method, dict):
            continue
        mask = method.get("refined", method.get("mask"))
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            continue
        if mask.shape != np.asarray(valid_mask).shape:
            continue
        summary = coverage_summary(mask, valid_mask)
        rows.append({"method": name, **summary})
    return rows


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    samples = np.asarray(values, dtype=np.float32).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {
            "count": 0, "mean": None, "median": None,
            "p10": None, "p90": None,
        }
    return {
        "count": int(samples.size),
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p90": float(np.percentile(samples, 90)),
    }


def _component_summary(mask: np.ndarray) -> dict[str, int | float | None]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = stats[1:count, cv2.CC_STAT_AREA]
    return {
        "region_count": int(sizes.size),
        "median_region_pixels": float(np.median(sizes)) if sizes.size else None,
        "largest_region_pixels": int(np.max(sizes)) if sizes.size else 0,
    }


def analyze_cloud_characteristics(
    selected_mask: np.ndarray,
    valid_mask: np.ndarray,
    features: dict[str, Any],
    thin_candidates: np.ndarray | None = None,
    dark_candidates: np.ndarray | None = None,
    ambiguous_mask: np.ndarray | None = None,
    spatial_grid_size: int = 8,
) -> dict[str, Any]:
    """Summarize actual extracted color, illumination, texture, and spatial data."""
    mask = np.asarray(selected_mask) > 0
    valid = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2 or valid.ndim != 2 or mask.shape != valid.shape:
        raise ValueError("Selected cloud mask and valid ROI mask must have matching 2D shapes.")
    if spatial_grid_size < 1:
        raise ValueError("Spatial grid size must be positive.")
    cloud = mask & valid
    sky = ~mask & valid

    illumination = features.get("illumination", {})
    texture = features.get("texture", {})
    brightness = np.asarray(illumination.get("brightness", np.empty((0, 0))))
    if brightness.shape != mask.shape:
        brightness = np.zeros(mask.shape, dtype=np.float32)
        brightness_available = False
    else:
        brightness_available = True

    brightness_histogram = {}
    bin_edges = np.linspace(0.0, 1.0, 13)
    if brightness_available:
        for label, selection in (("cloud", cloud), ("sky", sky)):
            brightness_histogram[label] = np.histogram(
                brightness[selection], bins=bin_edges
            )[0].tolist()
    else:
        brightness_histogram = {"cloud": [0] * 12, "sky": [0] * 12}

    rgb = np.asarray(features.get("rgb", np.empty((0, 0, 3))))
    hsv = np.asarray(features.get("hsv", np.empty((0, 0, 3))))
    color: dict[str, Any] = {"rgb": {}, "hsv": {}}
    color_bin_edges = np.linspace(0.0, 1.0, 9)
    for space, data, names, scales in (
        ("rgb", rgb, ("R", "G", "B"), (1.0, 1.0, 1.0)),
        ("hsv", hsv, ("H", "S", "V"), (180.0, 255.0, 255.0)),
    ):
        if data.shape != (*mask.shape, 3):
            continue
        for channel, name, scale in zip(range(3), names, scales):
            channel_values = data[:, :, channel].astype(np.float32) / scale
            color[space][name] = {
                "cloud": _distribution(channel_values[cloud]),
                "sky": _distribution(channel_values[sky]),
                "histogram": {
                    "bin_edges": color_bin_edges.tolist(),
                    "cloud": np.histogram(channel_values[cloud], bins=color_bin_edges)[0].tolist(),
                    "sky": np.histogram(channel_values[sky], bins=color_bin_edges)[0].tolist(),
                },
            }

    texture_summaries = {}
    for key in ("local_std", "local_entropy", "gradient"):
        if key in texture and np.asarray(texture[key]).shape == mask.shape:
            texture_summaries[key] = {
                "cloud": _distribution(texture[key][cloud]),
                "sky": _distribution(texture[key][sky]),
            }

    if "gradient" in texture and np.asarray(texture["gradient"]).shape == mask.shape:
        valid_gradients = np.asarray(texture["gradient"])[valid]
        edge_threshold = (
            float(np.percentile(valid_gradients, 75)) if valid_gradients.size else None
        )
        cloud_edge_pixels = (
            int(np.count_nonzero((texture["gradient"] >= edge_threshold) & cloud))
            if edge_threshold is not None and edge_threshold > 0 else 0
        )
    else:
        edge_threshold = None
        cloud_edge_pixels = 0

    height, width = mask.shape
    grid_height = min(spatial_grid_size, height)
    grid_width = min(spatial_grid_size, width)
    row_splits = np.array_split(np.arange(height), grid_height)
    column_splits = np.array_split(np.arange(width), grid_width)
    spatial_distribution = np.zeros((grid_height, grid_width), dtype=np.float32)
    for row_index, rows in enumerate(row_splits):
        for column_index, columns in enumerate(column_splits):
            cell_valid = valid[np.ix_(rows, columns)]
            denominator = int(np.count_nonzero(cell_valid))
            if denominator:
                cell_cloud = cloud[np.ix_(rows, columns)]
                spatial_distribution[row_index, column_index] = (
                    np.count_nonzero(cell_cloud) / denominator * 100.0
                )

    def candidate_summary(candidate: np.ndarray | None) -> dict[str, int | float | None]:
        if candidate is None:
            return {"pixel_count": 0, **_component_summary(np.zeros(mask.shape, dtype=np.uint8))}
        values = np.asarray(candidate)
        if values.shape != mask.shape:
            raise ValueError("Candidate and valid ROI dimensions must match.")
        selected = (values > 0) & valid
        return {
            "pixel_count": int(np.count_nonzero(selected)),
            **_component_summary(selected),
        }

    ambiguity = (
        np.zeros(mask.shape, dtype=bool)
        if ambiguous_mask is None
        else np.asarray(ambiguous_mask) > 0
    )
    if ambiguity.shape != mask.shape:
        raise ValueError("Ambiguous mask dimensions must match the selected mask.")

    return {
        "brightness_available": brightness_available,
        "brightness": {
            "cloud": _distribution(brightness[cloud]) if brightness_available else _distribution([]),
            "sky": _distribution(brightness[sky]) if brightness_available else _distribution([]),
            "histogram": brightness_histogram,
            "bin_edges": bin_edges.tolist(),
        },
        "color": color,
        "texture": texture_summaries,
        "gradient": {
            "edge_threshold_p75": edge_threshold,
            "cloud_edge_pixels": cloud_edge_pixels,
            "cloud": texture_summaries.get("gradient", {}).get("cloud", _distribution([])),
            "sky": texture_summaries.get("gradient", {}).get("sky", _distribution([])),
        },
        "spatial_distribution": spatial_distribution,
        "regions": _component_summary(cloud),
        "thin_candidates": candidate_summary(thin_candidates),
        "dark_candidates": candidate_summary(dark_candidates),
        "ambiguous_pixels": int(np.count_nonzero(ambiguity & valid)),
        "valid_roi_pixels": int(np.count_nonzero(valid)),
        "cloud_pixels": int(np.count_nonzero(cloud)),
        "sky_pixels": int(np.count_nonzero(sky)),
    }


def recommend_method_without_ground_truth(
    methods: dict[str, dict[str, Any]],
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    """Abstain from ranking methods without validated image-specific evidence."""
    rows = compare_method_coverage(methods, valid_mask)
    names = {row["method"] for row in rows}
    baseline = methods.get("Structure-aware", {})
    fusion = methods.get("Adaptive fusion", {})
    baseline_mask = baseline.get("refined")
    fusion_mask = fusion.get("refined")
    preserves_baseline = (
        isinstance(baseline_mask, np.ndarray)
        and isinstance(fusion_mask, np.ndarray)
        and baseline_mask.shape == fusion_mask.shape == np.asarray(valid_mask).shape
        and not np.any((baseline_mask > 0) & (fusion_mask == 0) & valid_mask)
    )
    signals = {
        "available_methods": len(rows),
        "thin_candidates_available": "Thin-cloud enhancement" in names,
        "dark_candidates_available": "Dark-cloud enhancement" in names,
        "fusion_preserves_baseline": preserves_baseline,
        "fusion_ambiguous_pixels": int(
            np.count_nonzero(
                (np.asarray(fusion.get("ambiguous", np.zeros_like(valid_mask))) > 0)
                & np.asarray(valid_mask, dtype=bool)
            )
        ),
    }
    return {
        "method": None,
        "message": "Insufficient evidence to determine a recommended method.",
        "reason": (
            "Internal scores and candidate masks are heuristic and have no ground-truth "
            "validation for this image. Review the baseline, thin/dark candidate masks, "
            "ambiguity, and fragmentation visually; coverage alone is not an accuracy measure."
        ),
        "signals": signals,
    }


def rank_methods_by_metric(
    metrics_by_method: dict[str, dict[str, float | int]],
    metric: str = "f1",
) -> dict[str, Any]:
    """Return all methods tied for the maximum observed per-image metric."""
    if metric not in {"iou", "precision", "recall", "f1", "accuracy"}:
        raise ValueError(f"Unsupported evaluation metric: {metric}")
    scores = {
        name: float(values[metric])
        for name, values in metrics_by_method.items()
        if metric in values and np.isfinite(values[metric])
    }
    if not scores:
        return {"metric": metric, "score": None, "methods": []}
    best_score = max(scores.values())
    return {
        "metric": metric,
        "score": best_score,
        "methods": [name for name, score in scores.items() if score == best_score],
    }
