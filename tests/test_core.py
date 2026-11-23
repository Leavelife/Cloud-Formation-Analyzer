import numpy as np
import pytest

from core.acquisition import bgr_from_rgb, rgb_from_bgr
from core.evaluation import calculate_metrics, make_comparison_map, prepare_ground_truth
from core.morphology import apply_morphology
from core.pipeline import process_image
from core.roi import build_valid_sky_mask, validate_roi
from core.segmentation import hsv_segmentation, lab_segmentation


def test_rgb_bgr_conversion_is_consistent():
    rgb = np.array([[[255, 0, 10], [20, 30, 40]]], dtype=np.uint8)
    bgr = bgr_from_rgb(rgb)
    assert bgr.tolist()[0][0] == [10, 0, 255]
    assert np.array_equal(rgb_from_bgr(bgr), rgb)


def test_validate_roi_clamps_and_rejects_empty():
    shape = (100, 200, 3)
    assert validate_roi((-20, 10, 250, 90), shape) == (0, 10, 200, 90)
    with pytest.raises(ValueError):
        validate_roi((20, 20, 20, 80), shape)


def test_polygon_valid_mask_excludes_pixels_outside_polygon():
    shape = (10, 10, 3)
    roi = (0, 0, 10, 10)
    valid = build_valid_sky_mask(shape, roi, [(2, 2), (8, 2), (8, 8), (2, 8)])
    assert valid.shape == (10, 10)
    assert not valid[0, 0]
    assert valid[5, 5]


def test_hsv_segmentation_separates_bright_neutral_cloud():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:10, :10] = [50, 100, 180]
    image[10:, 10:] = [245, 245, 245]
    params = {"h_min": 0, "h_max": 180, "s_min": 0, "s_max": 85,
              "v_min": 150, "v_max": 255}
    mask = hsv_segmentation(image, params)
    assert np.count_nonzero(mask[10:, 10:] > 0) == 100
    assert np.count_nonzero(mask[:10, :10] > 0) == 0


def test_lab_segmentation_separates_white_from_blue():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :10] = [235, 235, 235]
    image[:, 10:] = [30, 100, 220]
    mask = lab_segmentation(image, {"lab_l_min": 170, "lab_chroma_max": 28})
    assert np.count_nonzero(mask[:, :10]) == 100
    assert np.count_nonzero(mask[:, 10:]) == 0


def test_morphology_opening_removes_small_isolated_pixel():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255
    opened = apply_morphology(mask, "Opening", kernel_size=3)
    assert np.count_nonzero(opened) == 0


def test_morphology_erosion_iteration_count_is_not_applied_twice():
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[3:12, 3:12] = 255
    once = apply_morphology(mask, "Erosion", kernel_size=3, erosion_size=1)
    expected = apply_morphology(mask, "None", kernel_size=3, erosion_size=1)
    assert np.array_equal(once, expected)


def test_coverage_uses_valid_sky_pixels_only():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :] = [40, 100, 180]
    image[:, 5:] = [240, 240, 240]
    result = process_image(
        image,
        {
            "h_min": 0, "h_max": 180, "s_min": 0, "s_max": 85,
            "v_min": 150, "v_max": 255,
            "lab_l_min": 170, "lab_chroma_max": 28,
            "blur": 1, "method": "HSV", "morph": "None", "kernel": 3,
        },
        sky_roi=(0, 0, 10, 10),
    )
    assert result["sky_pixels"] == 100
    assert result["cloud_pixels"] == 50
    assert result["coverage"] == 50.0
    assert result["clear"] == 50.0


def test_empty_roi_validation():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        process_image(image, {}, sky_roi=(5, 5, 5, 8))


def test_evaluation_metrics_are_mathematically_correct():
    truth = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
    pred = np.array([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    metrics = calculate_metrics(pred, truth)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 2
    assert metrics["tn"] == 4
    assert metrics["iou"] == pytest.approx(25.0)
    assert metrics["precision"] == pytest.approx(50.0)
    assert metrics["recall"] == pytest.approx(1 / 3 * 100)
    assert metrics["f1"] == pytest.approx(40.0)
    assert metrics["accuracy"] == pytest.approx(62.5)


def test_ground_truth_dimensions_must_match_exactly():
    full_gray = np.full((8, 8), 255, dtype=np.uint8)
    prepared = prepare_ground_truth(full_gray[:, :, None].repeat(3, axis=2), (8, 8))
    assert prepared.shape == (8, 8)
    assert np.all(prepared == 1)
    with pytest.raises(ValueError):
        prepare_ground_truth(full_gray[:, :, None].repeat(3, axis=2), (7, 8))


def test_confusion_comparison_colors():
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    truth = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    comparison = make_comparison_map(pred, truth)
    assert comparison[0, 0].tolist() == [46, 160, 90]
    assert comparison[0, 1].tolist() == [225, 75, 75]
    assert comparison[1, 0].tolist() == [65, 135, 220]
