import numpy as np
import pytest

from core.analysis import (
    analyze_cloud_characteristics,
    compare_method_coverage,
    coverage_summary,
    rank_methods_by_metric,
    recommend_method_without_ground_truth,
)
from core.evaluation import evaluate_methods, prepare_ground_truth


def test_coverage_uses_valid_roi_and_cloud_sky_sum_to_one_hundred():
    mask = np.array([[255, 0, 255], [0, 255, 0]], dtype=np.uint8)
    valid = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
    summary = coverage_summary(mask, valid)
    assert summary["cloud_pixels"] == 2
    assert summary["sky_pixels"] == 3
    assert summary["valid_roi_pixels"] == 5
    assert summary["cloud_coverage"] + summary["sky_coverage"] == pytest.approx(100.0)


def test_method_comparison_skips_missing_or_misaligned_masks():
    valid = np.ones((2, 3), dtype=bool)
    methods = {
        "HSV": {"mask": np.array([[255, 0, 0], [0, 0, 0]], dtype=np.uint8)},
        "Bad dimensions": {"refined": np.zeros((3, 3), dtype=np.uint8)},
        "No mask": {},
    }
    rows = compare_method_coverage(methods, valid)
    assert [row["method"] for row in rows] == ["HSV"]
    assert rows[0]["cloud_pixels"] == 1


def test_characteristics_use_extracted_features_and_candidate_masks():
    mask = np.array([[255, 255, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    valid = np.ones(mask.shape, dtype=bool)
    features = {
        "illumination": {"brightness": np.array([[0.8, 0.6, 0.2, 0.3], [0.2, 0.3, 0.4, 0.5]])},
        "rgb": np.full((2, 4, 3), 0.5, dtype=np.float32),
        "hsv": np.full((2, 4, 3), 100, dtype=np.uint8),
        "texture": {
            "local_std": np.arange(8, dtype=np.float32).reshape(2, 4),
            "local_entropy": np.ones((2, 4), dtype=np.float32),
            "gradient": np.arange(8, dtype=np.float32).reshape(2, 4),
        },
    }
    summary = analyze_cloud_characteristics(
        mask,
        valid,
        features,
        thin_candidates=np.array([[0, 255, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
        dark_candidates=np.array([[0, 0, 0, 0], [0, 255, 0, 0]], dtype=np.uint8),
        ambiguous_mask=np.array([[0, 0, 255, 0], [0, 0, 0, 0]], dtype=np.uint8),
        spatial_grid_size=2,
    )
    assert summary["brightness"]["cloud"]["mean"] == pytest.approx(0.7)
    assert sum(summary["brightness"]["histogram"]["cloud"]) == 2
    assert sum(summary["color"]["rgb"]["R"]["histogram"]["cloud"]) == 2
    assert sum(summary["color"]["rgb"]["R"]["histogram"]["sky"]) == 6
    assert summary["regions"]["region_count"] == 1
    assert summary["thin_candidates"]["pixel_count"] == 1
    assert summary["dark_candidates"]["pixel_count"] == 1
    assert summary["ambiguous_pixels"] == 1
    assert summary["spatial_distribution"].shape == (2, 2)


def test_no_ground_truth_recommendation_abstains_and_explains_heuristic_limits():
    valid = np.ones((2, 2), dtype=bool)
    methods = {
        "Structure-aware": {"refined": np.array([[255, 0], [0, 0]], dtype=np.uint8)},
        "Thin-cloud enhancement": {"refined": np.array([[255, 255], [0, 0]], dtype=np.uint8)},
        "Dark-cloud enhancement": {"refined": np.array([[255, 0], [0, 0]], dtype=np.uint8)},
        "Adaptive fusion": {
            "refined": np.array([[255, 255], [0, 0]], dtype=np.uint8),
            "ambiguous": np.array([[0, 0], [255, 0]], dtype=np.uint8),
        },
    }
    recommendation = recommend_method_without_ground_truth(methods, valid)
    assert recommendation["method"] is None
    assert recommendation["message"] == "Insufficient evidence to determine a recommended method."
    assert recommendation["signals"]["fusion_preserves_baseline"]
    assert recommendation["signals"]["thin_candidates_available"]
    assert recommendation["signals"]["dark_candidates_available"]


def test_metric_ranking_returns_selected_metric_winner_and_ties():
    scores = {
        "HSV": {"f1": 80.0, "iou": 70.0},
        "Fusion": {"f1": 90.0, "iou": 75.0},
        "ARGD": {"f1": 90.0, "iou": 68.0},
    }
    assert rank_methods_by_metric(scores)["methods"] == ["Fusion", "ARGD"]
    assert rank_methods_by_metric(scores, "iou")["methods"] == ["Fusion"]
    assert rank_methods_by_metric({}, "precision")["methods"] == []
    with pytest.raises(ValueError):
        rank_methods_by_metric(scores, "coverage")


def test_evaluation_validates_binary_labels_and_scores_same_valid_roi():
    valid = np.array([[True, True], [False, True]])
    truth = prepare_ground_truth(
        np.array([[255, 0], [0, 255]], dtype=np.uint8), (2, 2)
    )
    methods = {
        "A": {"refined": np.array([[255, 0], [255, 255]], dtype=np.uint8)},
        "B": {"mask": np.array([[0, 0], [255, 255]], dtype=np.uint8)},
    }
    metrics = evaluate_methods(methods, truth, valid)
    assert set(metrics) == {"A", "B"}
    assert metrics["A"]["tp"] == 2
    assert metrics["A"]["fp"] == 0
    assert metrics["A"]["fn"] == 0
    assert metrics["A"]["tn"] == 1
    with pytest.raises(ValueError, match="binary"):
        prepare_ground_truth(np.array([[0, 128], [0, 255]], dtype=np.uint8), (2, 2))
    with pytest.raises(ValueError, match="do not match"):
        prepare_ground_truth(np.zeros((3, 2), dtype=np.uint8), (2, 2))
