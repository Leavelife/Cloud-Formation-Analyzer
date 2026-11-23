"""Hybrid thresholding for normalized blue/red ratio features."""

from __future__ import annotations

import numpy as np


def histogram_modality(
    feature: np.ndarray,
    std_threshold: float = 0.12,
) -> tuple[str, float, np.ndarray, np.ndarray]:
    """Classify feature distribution by normalized standard deviation.

    HYTA's published abstract describes a standard-deviation-based
    unimodal/bimodal decision. This explicit configurable dispersion criterion
    is a research-inspired implementation, not a claim of exact reproduction.
    """
    values = np.asarray(feature, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot analyze an empty feature map.")
    if not np.isfinite(std_threshold) or std_threshold < 0:
        raise ValueError("Standard-deviation boundary must be finite and non-negative.")
    deviation = float(np.std(values))
    hist, edges = np.histogram(values, bins=256, range=(-1.0, 1.0))
    modality = "bimodal" if deviation >= float(std_threshold) else "unimodal"
    return modality, deviation, hist, edges


def minimum_cross_entropy_threshold(
    feature: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """Find a Li-style iterative minimum cross-entropy threshold.

    The nonnegative histogram domain is obtained by mapping the bounded
    normalized ratio [-1, 1] to [0, 1]. Degenerate feature maps are rejected.
    """
    values = np.asarray(feature, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot threshold an empty feature map.")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("MCE iteration limit and tolerance must be positive.")
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-10:
        return (lo + hi) / 2.0
    scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    histogram, edges = np.histogram(scaled, bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    threshold = float(np.median(scaled))

    for _ in range(max_iterations):
        lower = scaled <= threshold
        upper = ~lower
        if not lower.any() or not upper.any():
            return lo + threshold * (hi - lo)
        mean_lower = float(np.mean(scaled[lower]))
        mean_upper = float(np.mean(scaled[upper]))
        if mean_lower <= 0:
            mean_lower = float(np.finfo(np.float64).eps)
        if mean_upper <= 0:
            mean_upper = float(np.finfo(np.float64).eps)
        denominator = np.log(mean_upper) - np.log(mean_lower)
        if abs(denominator) < 1e-12:
            break
        updated = (mean_upper - mean_lower) / denominator
        updated = float(np.clip(updated, 0.0, 1.0))
        if abs(updated - threshold) <= tolerance:
            threshold = updated
            break
        threshold = updated
    del histogram, centers
    return lo + threshold * (hi - lo)


def hyta_threshold(
    feature: np.ndarray,
    fixed_threshold: float = 0.0,
    std_threshold: float = 0.12,
) -> tuple[float, dict]:
    """Select fixed threshold for low dispersion, MCE for higher dispersion."""
    values = np.asarray(feature, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot threshold an empty feature map.")
    if not np.isfinite(fixed_threshold) or not -1.0 <= fixed_threshold <= 1.0:
        raise ValueError("Fixed HYTA threshold must be in [-1, 1].")
    modality, deviation, histogram, edges = histogram_modality(
        finite_values, std_threshold=std_threshold
    )
    threshold = (
        minimum_cross_entropy_threshold(finite_values)
        if modality == "bimodal"
        else float(fixed_threshold)
    )
    return threshold, {
        "modality": modality,
        "standard_deviation": deviation,
        "threshold": threshold,
        "histogram": histogram,
        "bin_edges": edges,
        "strategy": "minimum_cross_entropy" if modality == "bimodal" else "fixed",
    }
