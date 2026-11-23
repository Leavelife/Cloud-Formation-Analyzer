
"""Main image-processing pipeline."""

from __future__ import annotations

import cv2
import numpy as np
from time import perf_counter

from core.morphology import apply_morphology
from core.roi import build_valid_sky_mask, validate_roi
from core.features.argd import extract_argd
from core.features.nsv import extract_nsv
from core.features.normalized_ratio import extract_normalized_blue_red
from core.illumination import illumination_analysis
from core.segmentation import hsv_segmentation, lab_segmentation
from core.segmentation.hybrid import hybrid_segmentation
from core.segmentation.hyta import hyta_threshold
from core.segmentation.superpixel import create_superpixel_labels, segment_superpixels
from core.segmentation.structure_aware import (
    adaptive_fusion,
    classify_superpixels,
    dark_cloud_segmentation,
    thin_cloud_segmentation,
)
from core.texture import extract_texture


def preprocess_rgb(image_rgb: np.ndarray, blur_kernel: int = 1) -> np.ndarray:
    """Optionally smooth an RGB image before segmentation."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image.")
    blur_kernel = int(blur_kernel)
    if blur_kernel < 1:
        raise ValueError("Blur kernel must be at least 1.")
    if blur_kernel == 1:
        return image_rgb.copy()
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    return cv2.GaussianBlur(image_rgb, (blur_kernel, blur_kernel), 0)


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a blue highlight overlay while preserving the source image."""
    tint = np.zeros_like(image_rgb)
    tint[:] = (57, 169, 232)
    selected = mask > 0
    blended = cv2.addWeighted(image_rgb, 0.55, tint, 0.45, 0)
    overlay = image_rgb.copy()
    overlay[selected] = blended[selected]
    return overlay


def process_image(
    image_rgb: np.ndarray,
    params: dict,
    sky_roi: tuple[int, int, int, int] | list[int] | None = None,
    polygon_points: list[tuple[int, int]] | None = None,
    sunlight_region: tuple[int, int, int, int] | None = None,
) -> dict:
    """
    Process the selected sky region and return all comparable masks.

    Coverage is computed against the valid-sky mask only. Polygon points, when
    supplied, are full-image coordinates and are intersected with the ROI.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB image with dtype uint8.")

    pipeline_start = perf_counter()
    height, width = image_rgb.shape[:2]
    roi = validate_roi(sky_roi, image_rgb.shape)
    x1, y1, x2, y2 = roi
    working = image_rgb[y1:y2, x1:x2].copy()
    valid_full = build_valid_sky_mask(image_rgb.shape, roi, polygon_points)
    valid_mask = valid_full[y1:y2, x1:x2]
    if not np.any(valid_mask):
        raise ValueError("No valid sky pixels remain in the selected region.")

    smooth = preprocess_rgb(working, int(params.get("blur", 1)))
    setup_time = perf_counter() - pipeline_start
    sunlight_mode = str(params.get("sunlight_mode", "Warm-bright + user region"))
    marked_region_active = sunlight_mode in ("User region", "Warm-bright + user region")
    active_sunlight_region = sunlight_region if marked_region_active else None
    feature_start = perf_counter()
    illumination = illumination_analysis(
        smooth,
        active_sunlight_region,
        valid_mask=valid_mask,
    )
    illumination_time = perf_counter() - feature_start
    feature_start = perf_counter()
    argd, argd_weights = extract_argd(
        smooth,
        k=float(params.get("argd_k", 0.9)),
        sunlight_mask=illumination["user_sunlight_region"],
        sunlight_k=float(params.get("argd_sunlight_k", 0.6)),
    )
    argd_feature_time = perf_counter() - feature_start
    feature_start = perf_counter()
    nsv = extract_nsv(smooth)
    nsv_feature_time = perf_counter() - feature_start
    feature_start = perf_counter()
    normalized_ratio = extract_normalized_blue_red(smooth)
    ratio_feature_time = perf_counter() - feature_start
    feature_start = perf_counter()
    texture = extract_texture(
        smooth,
        window_size=int(params.get("texture_window", 9)),
        entropy_bins=int(params.get("texture_bins", 16)),
        adaptive_threshold_scale=float(params.get("adaptive_threshold_scale", 0.25)),
    )
    texture_time = perf_counter() - feature_start

    hsv_start = perf_counter()
    hsv_mask = hsv_segmentation(smooth, params)
    hsv_image = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    hsv_feature = hsv_image[:, :, 1].astype(np.float32) / 255.0
    hsv_time = perf_counter() - hsv_start
    lab_start = perf_counter()
    lab_mask = lab_segmentation(smooth, params)
    lab_image = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    lab_feature = lab_image[:, :, 0].astype(np.float32) / 255.0
    lab_time = perf_counter() - lab_start
    otsu_start = perf_counter()
    grayscale = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    otsu_threshold, otsu_mask = cv2.threshold(
        grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_feature = grayscale.astype(np.float32) / 255.0
    otsu_time = perf_counter() - otsu_start
    for candidate in (hsv_mask, lab_mask, otsu_mask):
        candidate[~valid_mask] = 0

    method_results: dict[str, dict] = {}

    def add_method(
        name: str,
        feature: np.ndarray,
        mask: np.ndarray,
        threshold,
        elapsed: float,
        diagnostics: dict | None = None,
        ambiguous: np.ndarray | None = None,
        score: np.ndarray | None = None,
        apply_refinement: bool = True,
    ) -> None:
        refinement_start = perf_counter()
        candidate = np.where(mask > 0, 255, 0).astype(np.uint8)
        candidate[~valid_mask] = 0
        if apply_refinement:
            refined = apply_morphology(
                candidate,
                operation=str(params.get("morph", "None")),
                kernel_size=int(params.get("kernel", 3)),
                erosion_size=int(params.get("erosion", 0)),
                dilation_size=int(params.get("dilation", 0)),
            )
        else:
            refined = candidate.copy()
        refined[~valid_mask] = 0
        method_ambiguous = (
            np.asarray(ambiguous, dtype=np.uint8).copy()
            if ambiguous is not None
            else np.zeros_like(candidate)
        )
        method_ambiguous[~valid_mask] = 0
        method_score = None if score is None else np.asarray(score).copy()
        if method_score is not None:
            method_score[~valid_mask] = 0
        method_results[name] = {
            "feature": feature,
            "mask": candidate,
            "refined": refined,
            "overlay": make_overlay(working, refined),
            "threshold": threshold,
            "processing_ms": (elapsed + perf_counter() - refinement_start) * 1000.0,
            "diagnostics": diagnostics or {},
            "ambiguous": method_ambiguous,
            "score": method_score,
        }

    start = perf_counter()
    add_method("HSV", hsv_feature, hsv_mask, "configured H/S/V", setup_time + hsv_time + perf_counter() - start)

    start = perf_counter()
    add_method(
        "LAB", lab_feature, lab_mask,
        f'L >= {int(params.get("lab_l_min", 170))}; chroma <= {float(params.get("lab_chroma_max", 28))}',
        setup_time + lab_time + perf_counter() - start,
    )

    start = perf_counter()
    add_method(
        "Otsu", otsu_feature, otsu_mask, float(otsu_threshold),
        setup_time + otsu_time + perf_counter() - start,
    )

    argd_threshold = float(params.get("argd_threshold", -0.15))
    start = perf_counter()
    add_method(
        "ARGD", argd, np.where(argd >= argd_threshold, 255, 0),
        argd_threshold, setup_time + illumination_time + argd_feature_time + perf_counter() - start,
        {"k_base": float(params.get("argd_k", 0.9)),
         "k_sunlight": float(params.get("argd_sunlight_k", 0.6))},
    )

    nsv_threshold = float(params.get("nsv_threshold", -0.3))
    start = perf_counter()
    add_method(
        "NSV", nsv, np.where(nsv <= nsv_threshold, 255, 0),
        nsv_threshold, setup_time + nsv_feature_time + perf_counter() - start,
    )

    start = perf_counter()
    ratio_threshold, hyta_info = hyta_threshold(
        normalized_ratio[valid_mask],
        fixed_threshold=float(params.get("hyta_fixed_threshold", 0.0)),
        std_threshold=float(params.get("hyta_std_threshold", 0.12)),
    )
    hyta_mask = np.where(normalized_ratio <= ratio_threshold, 255, 0)
    add_method(
        "HYTA", normalized_ratio, hyta_mask, ratio_threshold,
        setup_time + ratio_feature_time + perf_counter() - start, hyta_info,
    )

    start = perf_counter()
    superpixel_start = perf_counter()
    superpixel_labels, superpixel_implementation = create_superpixel_labels(
        smooth,
        number_of_superpixels=int(params.get("superpixel_count", 400)),
        compactness=float(params.get("superpixel_compactness", 10.0)),
        segmentation_method=str(params.get("segmentation_method", "SLIC")),
    )
    superpixel_label_time = perf_counter() - superpixel_start
    local_method_start = perf_counter()
    local_mask, boundaries, local_info = segment_superpixels(
        smooth,
        nsv,
        valid_mask=valid_mask,
        number_of_superpixels=int(params.get("superpixel_count", 400)),
        compactness=float(params.get("superpixel_compactness", 10.0)),
        local_threshold_scale=float(params.get("local_threshold_scale", 0.25)),
        local_feature_ceiling=float(params.get("local_feature_ceiling", -0.65)),
        segmentation_method=str(params.get("segmentation_method", "SLIC")),
        labels=superpixel_labels,
        implementation=superpixel_implementation,
    )
    superpixel_compute_time = superpixel_label_time + perf_counter() - local_method_start
    local_mask[~valid_mask] = 0
    add_method(
        "Superpixel", nsv, local_mask,
        local_info["mean_local_threshold"], setup_time + nsv_feature_time + perf_counter() - start, local_info,
    )

    start = perf_counter()
    texture_slic = classify_superpixels(
        smooth, argd, nsv, normalized_ratio, texture, illumination, params,
        valid_mask=valid_mask, use_spatial_continuity=False,
        labels=superpixel_labels, implementation=superpixel_implementation,
    )
    add_method(
        "Texture + SLIC", texture_slic["score"], texture_slic["mask"],
        float(params.get("structure_threshold", 0.55)),
        setup_time + illumination_time + argd_feature_time + nsv_feature_time
        + ratio_feature_time + texture_time + superpixel_label_time + perf_counter() - start,
        texture_slic["diagnostics"],
        ambiguous=texture_slic["ambiguous"],
        score=texture_slic["score"],
    )

    start = perf_counter()
    structure_aware = classify_superpixels(
        smooth, argd, nsv, normalized_ratio, texture, illumination, params,
        valid_mask=valid_mask, use_spatial_continuity=True,
        labels=superpixel_labels, implementation=superpixel_implementation,
    )
    add_method(
        "Structure-aware", structure_aware["score"], structure_aware["mask"],
        float(params.get("structure_threshold", 0.55)),
        setup_time + illumination_time + argd_feature_time + nsv_feature_time
        + ratio_feature_time + texture_time + superpixel_label_time + perf_counter() - start,
        structure_aware["diagnostics"],
        ambiguous=structure_aware["ambiguous"],
        score=structure_aware["score"],
    )

    baseline_method = method_results["Structure-aware"]
    enhancement_start = perf_counter()
    thin_result = thin_cloud_segmentation(
        baseline_method["refined"],
        structure_aware["score"],
        nsv,
        normalized_ratio,
        texture,
        superpixel_labels,
        params,
        valid_mask,
    )
    add_method(
        "Thin-cloud enhancement",
        thin_result["score"],
        thin_result["mask"],
        float(params.get("thin_score_threshold", 0.48)),
        setup_time + texture_time + superpixel_label_time + perf_counter() - enhancement_start,
        thin_result["diagnostics"],
        ambiguous=thin_result["ambiguous"],
        score=thin_result["score"],
        apply_refinement=False,
    )

    enhancement_start = perf_counter()
    dark_result = dark_cloud_segmentation(
        smooth,
        baseline_method["refined"],
        structure_aware["score"],
        nsv,
        normalized_ratio,
        texture,
        superpixel_labels,
        params,
        valid_mask,
    )
    add_method(
        "Dark-cloud enhancement",
        dark_result["score"],
        dark_result["mask"],
        float(params.get("dark_region_threshold", 0.48)),
        setup_time + illumination_time + texture_time + superpixel_label_time
        + perf_counter() - enhancement_start,
        dark_result["diagnostics"],
        ambiguous=dark_result["ambiguous"],
        score=dark_result["score"],
        apply_refinement=False,
    )

    enhancement_start = perf_counter()
    fusion_result = adaptive_fusion(
        baseline_method["refined"],
        structure_aware["score"],
        thin_result,
        dark_result,
        valid_mask,
        params,
    )
    add_method(
        "Adaptive fusion",
        fusion_result["score"],
        fusion_result["mask"],
        float(params.get("fusion_threshold", 0.52)),
        perf_counter() - enhancement_start,
        fusion_result["diagnostics"],
        ambiguous=fusion_result["ambiguous"],
        score=fusion_result["score"],
        apply_refinement=False,
    )

    start = perf_counter()
    hybrid_mask, ambiguous_mask, hybrid_score = hybrid_segmentation(
        argd, nsv, normalized_ratio, local_mask, illumination, texture, params
    )
    add_method(
        "Hybrid", hybrid_score, hybrid_mask,
        float(params.get("hybrid_threshold", 0.55)),
        setup_time + illumination_time + argd_feature_time + nsv_feature_time
        + ratio_feature_time + texture_time + superpixel_compute_time
        + perf_counter() - start,
        {"weights": {key: params.get(key) for key in
                     ("hybrid_w_argd", "hybrid_w_nsv", "hybrid_w_ratio", "hybrid_w_local")},
         "warm_bright_candidate_pixels": int(np.count_nonzero(illumination["warm_bright_candidates"]))},
        ambiguous=ambiguous_mask,
        score=hybrid_score,
    )

    method = str(params.get("method", "Adaptive fusion"))
    if method not in method_results:
        raise ValueError(f"Unknown final segmentation method: {method}")
    selected = method_results[method]
    raw_mask = selected["mask"]
    cleaned = selected["refined"]
    cleaned[~valid_mask] = 0

    valid_pixels = int(np.count_nonzero(valid_mask))
    cloud_pixels = int(np.count_nonzero(cleaned > 0))
    if valid_pixels == 0:
        raise ValueError("No valid sky pixels remain in the selected region.")
    cloud_pixels = min(cloud_pixels, valid_pixels)
    coverage = (cloud_pixels / valid_pixels * 100.0)

    return {
        "source": working,
        "raw_mask": raw_mask,
        "hsv_mask": hsv_mask,
        "lab_mask": lab_mask,
        "otsu_mask": otsu_mask,
        "features": {
            "rgb": smooth.astype(np.float32) / 255.0,
            "hsv": hsv_image,
            "argd": argd,
            "argd_weights": argd_weights,
            "nsv": nsv,
            "normalized_blue_red": normalized_ratio,
            "illumination": illumination,
            "texture": texture,
            "superpixel_boundaries": boundaries,
            "structure_boundaries": structure_aware["boundaries"],
            "thin_candidates": thin_result["candidates"],
            "dark_candidates": dark_result["candidates"],
        },
        "methods": method_results,
        "selected_method": method,
        "mask": cleaned,
        "overlay": make_overlay(working, cleaned),
        "cloud_pixels": cloud_pixels,
        "sky_pixels": valid_pixels,
        "coverage": coverage,
        "clear": max(0.0, 100.0 - coverage),
        "roi": roi,
        "sunlight_region": active_sunlight_region,
        "params": dict(params),
        "valid_mask": valid_mask,
        "full_image_shape": (height, width),
    }
