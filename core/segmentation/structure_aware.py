"""Region-based cloud classification using color, texture, and adjacency evidence."""

from __future__ import annotations

import numpy as np
import cv2

from core.segmentation.superpixel import create_superpixel_labels


def _cloud_evidence(feature: np.ndarray, threshold: float, higher_is_cloud: bool) -> np.ndarray:
    signed = feature - threshold if higher_is_cloud else threshold - feature
    return 1.0 / (1.0 + np.exp(-np.clip(signed / 0.08, -30.0, 30.0)))


def _robust_unit_scale(feature: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(feature, dtype=np.float32)
    samples = values[valid_mask & np.isfinite(values)]
    if samples.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(samples, (5, 95))
    if high - low <= 1e-6:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _region_means(
    labels: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active = (labels >= 0) & valid_mask
    ids = labels[active].astype(np.int64)
    count = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    counts = np.bincount(ids, minlength=count).astype(np.float64)
    sums = np.bincount(ids, weights=values[active], minlength=count)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return means.astype(np.float32), counts


def _region_adjacency(labels: np.ndarray, valid_mask: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for first, second, valid in (
        (labels[:, :-1], labels[:, 1:], valid_mask[:, :-1] & valid_mask[:, 1:]),
        (labels[:-1, :], labels[1:, :], valid_mask[:-1, :] & valid_mask[1:, :]),
    ):
        adjacent = valid & (first >= 0) & (second >= 0) & (first != second)
        if np.any(adjacent):
            unique_pairs = np.unique(
                np.sort(np.stack((first[adjacent], second[adjacent]), axis=1), axis=1),
                axis=0,
            )
            pairs.update((int(a), int(b)) for a, b in unique_pairs)
    return pairs


def classify_superpixels(
    image_rgb: np.ndarray,
    argd: np.ndarray,
    nsv: np.ndarray,
    normalized_ratio: np.ndarray,
    texture: dict[str, np.ndarray],
    illumination: dict[str, np.ndarray],
    params: dict,
    valid_mask: np.ndarray | None = None,
    use_spatial_continuity: bool = True,
    labels: np.ndarray | None = None,
    implementation: str | None = None,
) -> dict:
    """Classify whole superpixels from weighted multi-feature evidence.

    The returned score is a heuristic evidence score, not a probability.
    """
    shape = image_rgb.shape[:2]
    feature_maps = (argd, nsv, normalized_ratio)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image.")
    if any(np.asarray(feature).shape != shape for feature in feature_maps):
        raise ValueError("Color feature maps must match the RGB image dimensions.")
    required_texture = ("local_std", "local_entropy", "gradient")
    if any(key not in texture or np.asarray(texture[key]).shape != shape for key in required_texture):
        raise ValueError("All texture maps must match the RGB image dimensions.")
    if valid_mask is None:
        valid_mask = np.ones(shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != shape:
            raise ValueError("Valid-sky mask must match the RGB image dimensions.")
    if not valid_mask.any():
        raise ValueError("No valid pixels available for superpixel classification.")

    threshold = float(params.get("structure_threshold", 0.55))
    ambiguity_width = float(params.get("ambiguity_width", 0.12))
    if not 0.0 <= threshold <= 1.0 or ambiguity_width < 0:
        raise ValueError("Classification threshold must be in [0, 1] and ambiguity width non-negative.")
    weights = {
        "argd": float(params.get("structure_w_argd", 0.30)),
        "nsv": float(params.get("structure_w_nsv", 0.25)),
        "ratio": float(params.get("structure_w_ratio", 0.25)),
        "texture": float(params.get("structure_w_texture", 0.20)),
    }
    if any(weight < 0 or not np.isfinite(weight) for weight in weights.values()):
        raise ValueError("Feature weights must be finite and non-negative.")
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("At least one feature weight must be positive.")

    texture_score = (
        0.4 * _robust_unit_scale(texture["local_std"], valid_mask)
        + 0.3 * _robust_unit_scale(texture["local_entropy"], valid_mask)
        + 0.3 * _robust_unit_scale(texture["gradient"], valid_mask)
    )
    evidence = {
        "argd": _cloud_evidence(
            np.asarray(argd, dtype=np.float32),
            float(params.get("argd_threshold", -0.15)),
            True,
        ),
        "nsv": _cloud_evidence(
            np.asarray(nsv, dtype=np.float32),
            float(params.get("nsv_threshold", -0.3)),
            False,
        ),
        "ratio": _cloud_evidence(
            np.asarray(normalized_ratio, dtype=np.float32),
            float(params.get("ratio_threshold", 0.0)),
            False,
        ),
        "texture": texture_score,
    }
    pixel_score = sum(evidence[name] * weights[name] for name in weights) / weight_total
    if not np.isfinite(pixel_score[valid_mask]).all():
        raise ValueError("Feature evidence contains non-finite values.")

    if labels is None:
        labels, implementation = create_superpixel_labels(
            image_rgb,
            number_of_superpixels=int(params.get("superpixel_count", 400)),
            compactness=float(params.get("superpixel_compactness", 10.0)),
            segmentation_method=str(params.get("segmentation_method", "SLIC")),
        )
    elif labels.shape != shape:
        raise ValueError("Superpixel labels must match the RGB image dimensions.")
    else:
        implementation = implementation or str(params.get("segmentation_method", "SLIC"))
    region_scores, region_counts = _region_means(labels, pixel_score, valid_mask)
    region_texture, _ = _region_means(labels, texture_score, valid_mask)

    warm_bright = np.asarray(illumination.get("warm_bright_candidates", np.zeros(shape)), dtype=bool)
    user_sunlight = np.asarray(illumination.get("user_sunlight_region", np.zeros(shape)), dtype=bool)
    if warm_bright.shape != shape or user_sunlight.shape != shape:
        raise ValueError("Illumination masks must match the RGB image dimensions.")
    mode = str(params.get("sunlight_mode", "Warm-bright + user region"))
    if mode == "Warm-bright cue":
        interference = warm_bright
    elif mode == "User region":
        interference = user_sunlight
    elif mode == "Warm-bright + user region":
        interference = warm_bright | user_sunlight
    else:
        interference = np.zeros(shape, dtype=bool)
    region_interference, _ = _region_means(labels, interference.astype(np.float32), valid_mask)
    sunlight_penalty = float(params.get("sunlight_penalty", 0.35))
    if not 0.0 <= sunlight_penalty <= 1.0:
        raise ValueError("Sunlight penalty must be in [0, 1].")
    region_scores *= 1.0 - sunlight_penalty * region_interference

    initial_cloud = region_scores >= threshold
    promoted = np.zeros(region_scores.shape, dtype=bool)
    if use_spatial_continuity:
        continuity_margin = float(params.get("continuity_margin", 0.12))
        continuity_bonus = float(params.get("continuity_bonus", 0.08))
        if continuity_margin < 0 or continuity_bonus < 0:
            raise ValueError("Spatial continuity settings must be non-negative.")
        adjacent_to_cloud = np.zeros(region_scores.shape, dtype=bool)
        for first, second in _region_adjacency(labels, valid_mask):
            if initial_cloud[first]:
                adjacent_to_cloud[second] = True
            if initial_cloud[second]:
                adjacent_to_cloud[first] = True
        eligible = (
            (region_scores < threshold)
            & (region_scores >= threshold - continuity_margin)
            & adjacent_to_cloud
            & (region_texture >= 0.20)
        )
        region_scores[eligible] = np.minimum(
            1.0, region_scores[eligible] + continuity_bonus
        )
        promoted = eligible & (region_scores >= threshold)

    region_cloud = region_scores >= threshold
    score_map = np.zeros(shape, dtype=np.float32)
    score_map[labels >= 0] = region_scores[labels[labels >= 0]]
    cloud_mask = np.zeros(shape, dtype=np.uint8)
    assigned = (labels >= 0) & valid_mask
    cloud_mask[assigned] = region_cloud[labels[assigned]].astype(np.uint8) * 255
    ambiguous = (
        (np.abs(score_map - threshold) <= ambiguity_width) & assigned
    ).astype(np.uint8) * 255
    boundaries = np.zeros(shape, dtype=np.uint8)
    boundaries[1:, :] |= (labels[1:, :] != labels[:-1, :]).astype(np.uint8) * 255
    boundaries[:, 1:] |= (labels[:, 1:] != labels[:, :-1]).astype(np.uint8) * 255
    boundaries[~valid_mask] = 0

    region_valid = region_counts > 0
    return {
        "mask": cloud_mask,
        "ambiguous": ambiguous,
        "score": score_map,
        "labels": labels,
        "boundaries": boundaries,
        "diagnostics": {
            "implementation": implementation,
            "region_count": int(np.count_nonzero(region_valid)),
            "cloud_region_count": int(np.count_nonzero(region_cloud & region_valid)),
            "continuity_promoted_regions": int(np.count_nonzero(promoted)),
            "mean_region_score": float(np.mean(region_scores[region_valid])),
            "heuristic_score_not_probability": True,
            "weights": weights,
        },
    }


def _texture_support(texture: dict[str, np.ndarray], valid_mask: np.ndarray) -> np.ndarray:
    return (
        0.30 * _robust_unit_scale(texture["local_std"], valid_mask)
        + 0.25 * _robust_unit_scale(texture["local_entropy"], valid_mask)
        + 0.25 * _robust_unit_scale(texture["gradient"], valid_mask)
        + 0.20 * _robust_unit_scale(np.abs(texture["local_contrast"]), valid_mask)
    )


def _validate_texture_shapes(
    texture: dict[str, np.ndarray],
    shape: tuple[int, int],
    required: tuple[str, ...],
) -> None:
    if any(key not in texture or np.asarray(texture[key]).shape != shape for key in required):
        raise ValueError("Texture feature maps must contain the requested maps with matching dimensions.")


def _region_maxima(
    labels: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    count = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    maxima = np.zeros(count, dtype=np.float32)
    active = (labels >= 0) & valid_mask
    np.maximum.at(maxima, labels[active], values[active])
    return maxima


def thin_cloud_segmentation(
    baseline_mask: np.ndarray,
    baseline_score: np.ndarray,
    nsv: np.ndarray,
    normalized_ratio: np.ndarray,
    texture: dict[str, np.ndarray],
    labels: np.ndarray,
    params: dict,
    valid_mask: np.ndarray,
) -> dict:
    """Recover locally contrasting, cloud-colored filament candidates."""
    shape = baseline_mask.shape
    if any(array.shape != shape for array in (baseline_score, nsv, normalized_ratio, labels, valid_mask)):
        raise ValueError("Thin-cloud inputs must have matching dimensions.")
    valid_mask = np.asarray(valid_mask, dtype=bool)
    _validate_texture_shapes(
        texture,
        shape,
        ("local_std", "local_entropy", "gradient", "local_contrast", "contrast_z", "adaptive_bright"),
    )

    contrast_threshold = float(params.get("thin_contrast_threshold", 0.20))
    score_threshold = float(params.get("thin_score_threshold", 0.48))
    color_floor = float(params.get("thin_color_floor", 0.28))
    if contrast_threshold < 0 or not 0 <= score_threshold <= 1 or not 0 <= color_floor <= 1:
        raise ValueError("Thin-cloud thresholds are outside their valid ranges.")

    color_support = (
        0.55 * _cloud_evidence(
            normalized_ratio, float(params.get("ratio_threshold", 0.0)), False
        )
        + 0.45 * _cloud_evidence(
            nsv, float(params.get("nsv_threshold", -0.3)), False
        )
    )
    texture_support = _texture_support(texture, valid_mask)
    contrast_support = 1.0 / (
        1.0 + np.exp(-np.clip(
            (texture["contrast_z"] - contrast_threshold) / 0.55, -30.0, 30.0
        ))
    )
    pixel_score = (
        0.30 * contrast_support
        + 0.30 * color_support
        + 0.25 * texture_support
        + 0.15 * np.clip(baseline_score, 0.0, 1.0)
    )

    region_color, region_counts = _region_means(labels, color_support, valid_mask)
    region_texture, _ = _region_means(labels, texture_support, valid_mask)
    region_baseline, _ = _region_means(labels, baseline_score, valid_mask)
    region_contrast = _region_maxima(
        labels, contrast_support, valid_mask
    )
    region_eligible = (
        (region_color >= color_floor)
        & (region_contrast >= 0.50)
        & ((region_texture >= 0.12) | (region_baseline >= 0.35))
    )
    assigned = (labels >= 0) & valid_mask
    supported_region = np.zeros(shape, dtype=bool)
    supported_region[assigned] = region_eligible[labels[assigned]]
    candidates = (
        supported_region
        & (texture["contrast_z"] >= contrast_threshold)
        & np.asarray(texture["adaptive_bright"], dtype=bool)
        & (color_support >= color_floor)
        & (pixel_score >= score_threshold)
        & valid_mask
    )
    mask = ((baseline_mask > 0) | candidates).astype(np.uint8) * 255
    ambiguity_width = float(params.get("ambiguity_width", 0.12))
    ambiguous = (
        (np.abs(pixel_score - score_threshold) <= ambiguity_width)
        & supported_region
        & valid_mask
    ).astype(np.uint8) * 255
    return {
        "mask": mask,
        "candidates": candidates.astype(np.uint8) * 255,
        "score": np.where(valid_mask, pixel_score, 0.0).astype(np.float32),
        "ambiguous": ambiguous,
        "diagnostics": {
            "candidate_pixels": int(np.count_nonzero(candidates)),
            "eligible_regions": int(np.count_nonzero(region_eligible & (region_counts > 0))),
            "threshold": score_threshold,
            "interpretation": "Heuristic local-contrast and color evidence; not a probability.",
        },
    }


def dark_cloud_segmentation(
    image_rgb: np.ndarray,
    baseline_mask: np.ndarray,
    baseline_score: np.ndarray,
    nsv: np.ndarray,
    normalized_ratio: np.ndarray,
    texture: dict[str, np.ndarray],
    labels: np.ndarray,
    params: dict,
    valid_mask: np.ndarray,
) -> dict:
    """Recover textured, neutral dark regions with local or neighboring support."""
    shape = image_rgb.shape[:2]
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image.")
    if any(array.shape != shape for array in (
        baseline_mask, baseline_score, nsv, normalized_ratio, labels, valid_mask,
    )):
        raise ValueError("Dark-cloud inputs must have matching dimensions.")
    valid_mask = np.asarray(valid_mask, dtype=bool)
    _validate_texture_shapes(
        texture,
        shape,
        ("local_std", "local_entropy", "gradient", "local_contrast", "contrast_z"),
    )

    region_threshold = float(params.get("dark_region_threshold", 0.48))
    saturation_ceiling = float(params.get("dark_saturation_ceiling", 0.48))
    texture_floor = float(params.get("dark_texture_floor", 0.16))
    continuity_bonus = float(params.get("dark_continuity_bonus", 0.10))
    if not 0 <= region_threshold <= 1 or not 0 <= saturation_ceiling <= 1:
        raise ValueError("Dark-cloud thresholds must be in [0, 1].")
    if min(texture_floor, continuity_bonus) < 0:
        raise ValueError("Dark-cloud texture and continuity settings must be non-negative.")

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    low_saturation = 1.0 / (
        1.0 + np.exp(-np.clip((saturation_ceiling - saturation) / 0.10, -30.0, 30.0))
    )
    neutral_color = (
        0.55 * _cloud_evidence(
            normalized_ratio, float(params.get("ratio_threshold", 0.0)), False
        )
        + 0.45 * _cloud_evidence(
            nsv, float(params.get("nsv_threshold", -0.3)), False
        )
    )
    texture_support = _texture_support(texture, valid_mask)
    edge_support = _robust_unit_scale(texture["gradient"], valid_mask)
    relative_darkness = 1.0 / (
        1.0 + np.exp(-np.clip(-texture["contrast_z"] / 0.60, -30.0, 30.0))
    )
    pixel_score = (
        0.30 * neutral_color
        + 0.20 * low_saturation
        + 0.20 * texture_support
        + 0.15 * edge_support
        + 0.10 * relative_darkness
        + 0.05 * np.clip(baseline_score, 0.0, 1.0)
    )

    region_color, region_counts = _region_means(labels, neutral_color, valid_mask)
    region_saturation, _ = _region_means(labels, low_saturation, valid_mask)
    region_texture, _ = _region_means(labels, texture_support, valid_mask)
    region_edges, _ = _region_means(labels, edge_support, valid_mask)
    region_scores, _ = _region_means(labels, pixel_score, valid_mask)
    region_baseline, _ = _region_means(labels, (baseline_mask > 0).astype(np.float32), valid_mask)
    adjacent_to_baseline = np.zeros(region_counts.shape, dtype=bool)
    baseline_regions = region_baseline >= 0.10
    for first, second in _region_adjacency(labels, valid_mask):
        if baseline_regions[first]:
            adjacent_to_baseline[second] = True
        if baseline_regions[second]:
            adjacent_to_baseline[first] = True

    structural_support = (
        (region_texture >= texture_floor)
        | (region_edges >= texture_floor)
        | adjacent_to_baseline
    )
    region_scores = np.minimum(
        1.0,
        region_scores + adjacent_to_baseline.astype(np.float32) * continuity_bonus,
    )
    eligible_regions = (
        (region_color >= 0.40)
        & (region_saturation >= 0.50)
        & structural_support
        & (region_scores >= region_threshold)
        & (region_counts > 0)
    )
    assigned = (labels >= 0) & valid_mask
    selected_regions = np.zeros(shape, dtype=bool)
    selected_regions[assigned] = eligible_regions[labels[assigned]]
    candidates = selected_regions & (pixel_score >= region_threshold - 0.12)
    mask = ((baseline_mask > 0) | candidates).astype(np.uint8) * 255
    score_map = np.zeros(shape, dtype=np.float32)
    score_map[labels >= 0] = region_scores[labels[labels >= 0]]
    ambiguity_width = float(params.get("ambiguity_width", 0.12))
    ambiguous = (
        (np.abs(score_map - region_threshold) <= ambiguity_width)
        & assigned
    ).astype(np.uint8) * 255
    return {
        "mask": mask,
        "candidates": candidates.astype(np.uint8) * 255,
        "score": score_map,
        "ambiguous": ambiguous,
        "diagnostics": {
            "candidate_pixels": int(np.count_nonzero(candidates)),
            "eligible_regions": int(np.count_nonzero(eligible_regions)),
            "continuity_supported_regions": int(np.count_nonzero(
                eligible_regions & adjacent_to_baseline
            )),
            "threshold": region_threshold,
            "interpretation": "Heuristic regional texture, color, and adjacency evidence; not a probability.",
        },
    }


def adaptive_fusion(
    baseline_mask: np.ndarray,
    baseline_score: np.ndarray,
    thin_result: dict,
    dark_result: dict,
    valid_mask: np.ndarray,
    params: dict,
) -> dict:
    """Preserve the existing mask and score-gate structure-supported additions."""
    shape = baseline_mask.shape
    arrays = (
        baseline_score,
        thin_result["score"],
        dark_result["score"],
        thin_result["candidates"],
        dark_result["candidates"],
        valid_mask,
    )
    if any(array.shape != shape for array in arrays):
        raise ValueError("Fusion inputs must have matching dimensions.")

    baseline_weight = float(params.get("fusion_w_baseline", 0.45))
    thin_weight = float(params.get("fusion_w_thin", 0.55))
    dark_weight = float(params.get("fusion_w_dark", 0.45))
    threshold = float(params.get("fusion_threshold", 0.52))
    if not 0 <= baseline_weight <= 1 or min(thin_weight, dark_weight) < 0:
        raise ValueError("Fusion weights must be non-negative and baseline weight in [0, 1].")
    if thin_weight + dark_weight <= 0 or not 0 <= threshold <= 1:
        raise ValueError("At least one enhancement weight must be positive and threshold in [0, 1].")

    thin_candidates = thin_result["candidates"] > 0
    dark_candidates = dark_result["candidates"] > 0
    active_weight = thin_candidates * thin_weight + dark_candidates * dark_weight
    enhancement_score = np.divide(
        thin_result["score"] * thin_candidates * thin_weight
        + dark_result["score"] * dark_candidates * dark_weight,
        active_weight,
        out=np.zeros(shape, dtype=np.float32),
        where=active_weight > 0,
    )
    has_enhancement = active_weight > 0
    score = np.where(
        has_enhancement,
        baseline_weight * baseline_score + (1.0 - baseline_weight) * enhancement_score,
        baseline_score,
    )
    accepted_additions = has_enhancement & (score >= threshold) & valid_mask
    final_mask = ((baseline_mask > 0) | accepted_additions).astype(np.uint8) * 255
    final_mask[~valid_mask] = 0
    ambiguity_width = float(params.get("ambiguity_width", 0.12))
    ambiguous = (
        (np.abs(score - threshold) <= ambiguity_width)
        & valid_mask
    ).astype(np.uint8) * 255
    score = np.where(valid_mask, score, 0.0).astype(np.float32)
    return {
        "mask": final_mask,
        "score": score,
        "ambiguous": ambiguous,
        "diagnostics": {
            "threshold": threshold,
            "baseline_preserved": True,
            "accepted_enhancement_pixels": int(np.count_nonzero(accepted_additions)),
            "heuristic_score_not_probability": True,
            "weights": {
                "baseline": baseline_weight,
                "thin": thin_weight,
                "dark": dark_weight,
            },
        },
    }
