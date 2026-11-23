import numpy as np
import pytest

from core.evaluation import calculate_metrics
from core.features.argd import extract_argd
from core.features.normalized_ratio import extract_normalized_blue_red
from core.features.nsv import extract_nsv
from core.pipeline import process_image
from core.segmentation.hybrid import hybrid_segmentation
from core.segmentation.hyta import (
    histogram_modality,
    hyta_threshold,
    minimum_cross_entropy_threshold,
)
from core.segmentation.superpixel import segment_superpixels
from core.segmentation.structure_aware import (
    adaptive_fusion,
    classify_superpixels,
    dark_cloud_segmentation,
    thin_cloud_segmentation,
)
from core.texture import extract_texture


def test_argd_uses_normalized_float_channels_and_region_specific_k():
    image = np.array([[[255, 64, 0], [255, 64, 0]]], dtype=np.uint8)
    sunlight = np.array([[False, True]])
    feature, weights = extract_argd(image, k=1.0, sunlight_mask=sunlight, sunlight_k=0.5)
    assert feature.dtype == np.float32
    assert feature[0, 0] == pytest.approx(1.0 - 64 / 255)
    assert feature[0, 1] == pytest.approx(0.5 - 64 / 255)
    assert weights.tolist() == [[1.0, 0.5]]


def test_nsv_is_normalized_difference_and_handles_black():
    image = np.array([[[255, 255, 255], [0, 0, 255], [0, 0, 0]]], dtype=np.uint8)
    feature = extract_nsv(image)
    assert feature[0, 0] == pytest.approx(-1.0, abs=1e-5)
    assert feature[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert np.isfinite(feature[0, 2])
    assert feature[0, 2] == 0.0


def test_normalized_blue_red_ratio_is_float_and_bounded():
    image = np.array([[[100, 0, 200], [0, 0, 0]]], dtype=np.uint8)
    ratio = extract_normalized_blue_red(image)
    assert ratio.dtype == np.float32
    assert ratio[0, 0] == pytest.approx(1 / 3, abs=1e-5)
    assert ratio[0, 1] == 0.0


def test_hyta_modality_and_mce_threshold_for_separated_modes():
    feature = np.concatenate([np.full(200, -0.8), np.full(200, 0.55)]).reshape(20, 20)
    modality, deviation, histogram, edges = histogram_modality(feature, std_threshold=0.1)
    threshold = minimum_cross_entropy_threshold(feature)
    selected, info = hyta_threshold(feature, fixed_threshold=-0.1, std_threshold=0.1)
    assert modality == "bimodal"
    assert deviation > 0.1
    assert histogram.sum() == feature.size
    assert len(edges) == 257
    assert -0.8 < threshold < 0.55
    assert selected == pytest.approx(threshold)
    assert info["strategy"] == "minimum_cross_entropy"


def test_hyta_constant_feature_uses_fixed_threshold_without_failure():
    feature = np.full((8, 8), 0.2, dtype=np.float32)
    threshold, info = hyta_threshold(feature, fixed_threshold=0.1, std_threshold=0.1)
    assert threshold == 0.1
    assert info["modality"] == "unimodal"
    with pytest.raises(ValueError):
        histogram_modality(np.array([]))


def test_superpixel_local_method_does_not_select_uniform_warm_clear_feature():
    image = np.full((40, 60, 3), [210, 150, 100], dtype=np.uint8)
    feature = np.full((40, 60), -0.35, dtype=np.float32)
    feature[15:25, 20:40] = -0.9
    mask, boundaries, diagnostics = segment_superpixels(
        image,
        feature,
        number_of_superpixels=24,
        compactness=10,
        local_feature_ceiling=-0.65,
    )
    assert diagnostics["region_count"] > 0
    assert mask.shape == feature.shape
    assert boundaries.shape == feature.shape
    assert np.count_nonzero(mask[feature == -0.35]) == 0
    assert np.count_nonzero(mask[feature == -0.9]) > 0
    assert diagnostics["implementation"] in ("SLIC", "OpenCV watershed fallback")


def test_texture_features_are_finite_and_respond_to_local_variation():
    rng = np.random.default_rng(42)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    texture = extract_texture(image, window_size=7)
    assert set(texture) == {
        "local_std", "local_entropy", "gradient",
        "local_contrast", "contrast_z", "adaptive_threshold", "adaptive_bright",
    }
    assert all(np.isfinite(feature).all() for feature in texture.values())
    assert texture["local_std"][16, 16] > texture["local_std"][2, 2]
    assert texture["local_entropy"][16, 16] > texture["local_entropy"][2, 2]


def test_local_contrast_and_adaptive_threshold_retain_a_thin_filament():
    image = np.full((32, 40, 3), [80, 145, 210], dtype=np.uint8)
    image[14:18, 5:35] = [135, 175, 215]
    texture = extract_texture(
        image, window_size=9, adaptive_threshold_scale=0.25
    )
    assert np.isfinite(texture["contrast_z"]).all()
    assert texture["contrast_z"][15, 20] > texture["contrast_z"][5, 20]
    assert texture["adaptive_bright"][15, 20]
    assert not texture["adaptive_bright"][5, 20]


def test_thin_enhancement_selects_supported_filament_not_uniform_sky():
    height, width = 12, 40
    labels = np.zeros((height, width), dtype=np.int32)
    labels[:, 36:] = 1
    baseline_mask = np.zeros((height, width), dtype=np.uint8)
    baseline_score = np.zeros((height, width), dtype=np.float32)
    ratio = np.full((height, width), 0.4, dtype=np.float32)
    nsv = np.full((height, width), -0.1, dtype=np.float32)
    ratio[:, 36:] = -0.1
    nsv[:, 36:] = -0.6
    texture = {
        "local_std": np.zeros((height, width), dtype=np.float32),
        "local_entropy": np.zeros((height, width), dtype=np.float32),
        "gradient": np.zeros((height, width), dtype=np.float32),
        "local_contrast": np.zeros((height, width), dtype=np.float32),
        "contrast_z": np.zeros((height, width), dtype=np.float32),
        "adaptive_threshold": np.zeros((height, width), dtype=np.float32),
        "adaptive_bright": np.zeros((height, width), dtype=bool),
    }
    for key in ("local_std", "local_entropy", "gradient", "local_contrast"):
        texture[key][:, 36:] = 1.0
    texture["contrast_z"][:, 36:] = 1.0
    texture["adaptive_bright"][:, 36:] = True
    result = thin_cloud_segmentation(
        baseline_mask,
        baseline_score,
        nsv,
        ratio,
        texture,
        labels,
        {"thin_contrast_threshold": 0.2, "thin_score_threshold": 0.48},
        np.ones((height, width), dtype=bool),
    )
    assert np.all(result["candidates"][:, 36:] == 255)
    assert np.count_nonzero(result["candidates"][:, :36]) == 0


def test_dark_enhancement_requires_texture_or_baseline_neighbor_support():
    height, width = 16, 45
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :15] = [220, 220, 220]
    image[:, 15:30] = [55, 57, 60]
    image[:, 30:] = [25, 40, 70]
    labels = np.tile(np.repeat([0, 1, 2], 15), (height, 1))
    baseline_mask = np.zeros((height, width), dtype=np.uint8)
    baseline_mask[:, :15] = 255
    baseline_score = np.zeros((height, width), dtype=np.float32)
    baseline_score[:, :15] = 1.0
    texture = extract_texture(image, window_size=5)
    nsv = extract_nsv(image)
    ratio = extract_normalized_blue_red(image)
    result = dark_cloud_segmentation(
        image,
        baseline_mask,
        baseline_score,
        nsv,
        ratio,
        texture,
        labels,
        {},
        np.ones((height, width), dtype=bool),
    )
    assert np.all(result["candidates"][:, 15:30] == 255)
    assert np.count_nonzero(result["candidates"][:, 30:]) == 0


def test_adaptive_fusion_preserves_baseline_and_score_gates_new_candidates():
    baseline_mask = np.array([[255, 0, 0, 0]], dtype=np.uint8)
    baseline_score = np.array([[0.9, 0.2, 0.3, 0.2]], dtype=np.float32)
    thin = {
        "candidates": np.array([[0, 255, 0, 255]], dtype=np.uint8),
        "score": np.array([[0.0, 0.9, 0.1, 0.4]], dtype=np.float32),
    }
    dark = {
        "candidates": np.array([[0, 0, 255, 255]], dtype=np.uint8),
        "score": np.array([[0.0, 0.1, 0.9, 0.6]], dtype=np.float32),
    }
    valid = np.array([[True, True, True, False]])
    result = adaptive_fusion(
        baseline_mask, baseline_score, thin, dark, valid, {}
    )
    assert result["mask"].tolist() == [[255, 255, 255, 0]]
    assert result["diagnostics"]["baseline_preserved"]
    assert result["diagnostics"]["accepted_enhancement_pixels"] == 2


def test_pipeline_respects_roi_polygon_for_each_method_and_coverage():
    image = np.full((30, 40, 3), [30, 90, 180], dtype=np.uint8)
    image[8:22, 10:30] = [245, 245, 245]
    valid_polygon = [(10, 10), (30, 10), (30, 20), (10, 20)]
    params = {
        "method": "Hybrid",
        "morph": "None",
        "kernel": 3,
        "superpixel_count": 25,
    }
    result = process_image(
        image,
        params,
        sky_roi=(5, 5, 35, 25),
        polygon_points=valid_polygon,
    )
    assert result["source"].shape == (20, 30, 3)
    assert result["sky_pixels"] == int(result["valid_mask"].sum())
    assert result["cloud_pixels"] <= result["sky_pixels"]
    assert np.all(result["mask"][~result["valid_mask"]] == 0)
    assert set(result["methods"]) == {
        "HSV", "LAB", "Otsu", "ARGD", "NSV", "HYTA", "Superpixel",
        "Texture + SLIC", "Structure-aware", "Thin-cloud enhancement",
        "Dark-cloud enhancement", "Adaptive fusion", "Hybrid",
    }
    assert all(method["mask"].shape == (20, 30) for method in result["methods"].values())
    assert result["methods"]["Structure-aware"]["diagnostics"]["implementation"] == "SLIC"
    assert result["features"]["rgb"].shape == (20, 30, 3)
    assert result["features"]["hsv"].shape == (20, 30, 3)
    assert np.all(result["methods"]["Structure-aware"]["mask"][~result["valid_mask"]] == 0)
    assert np.all(
        result["methods"]["Adaptive fusion"]["refined"][
            result["methods"]["Structure-aware"]["refined"] > 0
        ] == 255
    )
    for name in ("Thin-cloud enhancement", "Dark-cloud enhancement", "Adaptive fusion"):
        assert np.all(result["methods"][name]["refined"][~result["valid_mask"]] == 0)


def test_structure_aware_continuity_preserves_adjacent_dark_shadow_region():
    height, width = 12, 36
    labels = np.tile(np.repeat([0, 1, 2], width // 3), (height, 1))
    argd = np.zeros((height, width), dtype=np.float32)
    argd[:, :12] = 0.18
    argd[:, 12:24] = 0.0
    argd[:, 24:] = -0.12
    texture = {
        "local_std": np.zeros((height, width), dtype=np.float32),
        "local_entropy": np.zeros((height, width), dtype=np.float32),
        "gradient": np.zeros((height, width), dtype=np.float32),
    }
    texture["local_std"][:, :12] = 0.5
    texture["local_std"][:, 12:24] = 1.0
    zeros = np.zeros((height, width), dtype=np.float32)
    illumination = {
        "warm_bright_candidates": np.zeros((height, width), dtype=bool),
        "user_sunlight_region": np.zeros((height, width), dtype=bool),
    }
    params = {
        "argd_threshold": 0.0,
        "structure_threshold": 0.55,
        "structure_w_argd": 1.0,
        "structure_w_nsv": 0.0,
        "structure_w_ratio": 0.0,
        "structure_w_texture": 0.0,
        "sunlight_mode": "None",
        "continuity_margin": 0.12,
        "continuity_bonus": 0.08,
    }
    without_continuity = classify_superpixels(
        np.zeros((height, width, 3), dtype=np.uint8),
        argd, zeros, zeros, texture, illumination, params,
        labels=labels, implementation="test labels",
        use_spatial_continuity=False,
    )
    result = classify_superpixels(
        np.zeros((height, width, 3), dtype=np.uint8),
        argd, zeros, zeros, texture, illumination, params,
        labels=labels, implementation="test labels",
    )
    assert np.all(result["mask"][:, :12] == 255)
    assert np.all(without_continuity["mask"][:, 12:24] == 0)
    assert np.all(result["mask"][:, 12:24] == 255)
    assert np.all(result["mask"][:, 24:] == 0)
    assert result["diagnostics"]["continuity_promoted_regions"] == 1


def test_structure_aware_penalizes_uniform_warm_bright_sky_cue():
    image = np.full((32, 48, 3), [235, 160, 100], dtype=np.uint8)
    params = {
        "method": "Structure-aware",
        "morph": "None",
        "kernel": 1,
        "superpixel_count": 24,
        "sunlight_mode": "Warm-bright cue",
        "sunlight_penalty": 0.5,
    }
    result = process_image(image, params)
    assert np.count_nonzero(result["methods"]["Structure-aware"]["mask"]) == 0
    assert result["methods"]["Structure-aware"]["diagnostics"]["heuristic_score_not_probability"]


def test_pipeline_smoke_synthetic_cloud_and_illumination_conditions():
    height, width = 48, 64
    clear_blue = np.full((height, width, 3), [80, 145, 210], dtype=np.uint8)
    daylight = clear_blue.copy()
    daylight[12:38, 15:48] = [230, 232, 235]

    dark_cloud = clear_blue.copy()
    dark_cloud[12:38, 15:48] = [65, 70, 76]
    dark_cloud[15:35:2, 17:46:2] = [90, 94, 98]

    shadowed_cloud = daylight.copy()
    shadowed_cloud[22:34, 20:42] = [75, 80, 86]

    golden_cloud = np.full((height, width, 3), [218, 156, 100], dtype=np.uint8)
    golden_cloud[12:38, 15:48] = [244, 202, 158]
    golden_cloud[22:34, 20:42] = [80, 82, 86]

    warm_clear = np.full((height, width, 3), [235, 166, 103], dtype=np.uint8)
    sunset = np.full((height, width, 3), [225, 125, 78], dtype=np.uint8)
    sunset[8:40, 26:38] = [255, 215, 150]
    thin_cloud = clear_blue.copy()
    thin_cloud[24:27, 8:56] = [170, 195, 220]
    haze = np.full((height, width, 3), [154, 174, 190], dtype=np.uint8)
    haze[16:34, 18:46] = [166, 183, 197]

    synthetic_conditions = {
        "clear blue sky": clear_blue,
        "white daylight cloud": daylight,
        "dark gray cloud": dark_cloud,
        "strongly shadowed cloud": shadowed_cloud,
        "golden-hour orange cloud and dark shadow": golden_cloud,
        "warm orange clear sky": warm_clear,
        "sunset with direct bright region": sunset,
        "thin cloud": thin_cloud,
        "hazy sky": haze,
    }
    for condition, image in synthetic_conditions.items():
        result = process_image(
            image,
            {
                "method": "Structure-aware",
                "morph": "None",
                "kernel": 1,
                "superpixel_count": 24,
                "segmentation_method": "SLIC",
            },
        )
        method = result["methods"]["Structure-aware"]
        assert method["mask"].shape == (height, width), condition
        assert np.isfinite(method["score"]).all(), condition
        assert 0.0 <= result["coverage"] <= 100.0, condition
        assert np.count_nonzero(method["ambiguous"]) <= result["sky_pixels"], condition


def test_adaptive_fusion_adds_supported_thin_pixels_without_copying_every_candidate():
    image = np.full((48, 64, 3), [80, 145, 210], dtype=np.uint8)
    image[21:27, 8:56] = [135, 175, 215]
    result = process_image(
        image,
        {
            "method": "Adaptive fusion",
            "morph": "None",
            "superpixel_count": 24,
        },
    )
    baseline_pixels = np.count_nonzero(result["methods"]["Structure-aware"]["refined"])
    thin_pixels = np.count_nonzero(result["methods"]["Thin-cloud enhancement"]["refined"])
    fused_pixels = np.count_nonzero(result["methods"]["Adaptive fusion"]["refined"])
    assert baseline_pixels == 0
    assert 0 < fused_pixels < thin_pixels


def test_evaluation_excludes_invalid_pixels_from_confusion_counts():
    predicted = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    truth = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    valid = np.array([[1, 1], [0, 0]], dtype=bool)
    metrics = calculate_metrics(predicted, truth, valid_mask=valid)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["tn"] == 0
    assert metrics["accuracy"] == 50.0


def test_hybrid_sunlight_cue_reduces_warm_bright_false_cloud_evidence():
    shape = (4, 4)
    argd = np.full(shape, 0.2, dtype=np.float32)
    nsv = np.full(shape, -0.4, dtype=np.float32)
    ratio = np.full(shape, -0.4, dtype=np.float32)
    local = np.zeros(shape, dtype=np.uint8)
    texture = {"local_std": np.zeros(shape, dtype=np.float32)}
    illumination = {
        "warm_bright_candidates": np.ones(shape, dtype=bool),
        "user_sunlight_region": np.zeros(shape, dtype=bool),
    }
    params = {"sunlight_mode": "None", "hybrid_threshold": 0.55}
    baseline, _, baseline_score = hybrid_segmentation(
        argd, nsv, ratio, local, illumination, texture, params
    )
    params.update({"sunlight_mode": "Warm-bright cue", "sunlight_penalty": 0.5})
    penalized, ambiguous, penalized_score = hybrid_segmentation(
        argd, nsv, ratio, local, illumination, texture, params
    )
    assert np.all(penalized_score < baseline_score)
    assert np.all(baseline == 255)
    assert np.all(penalized == 0)
    assert np.all(ambiguous == 0)
