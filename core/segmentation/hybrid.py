"""Evidence-weighted hybrid segmentation with an explicit ambiguity map."""

from __future__ import annotations

import numpy as np


def _cloud_score(feature: np.ndarray, threshold: float, direction: str, scale: float) -> np.ndarray:
    signed = threshold - feature if direction == "lower" else feature - threshold
    return 1.0 / (1.0 + np.exp(-np.clip(signed / max(scale, 1e-6), -30, 30)))


def hybrid_segmentation(
    argd: np.ndarray,
    nsv: np.ndarray,
    normalized_ratio: np.ndarray,
    local_mask: np.ndarray,
    illumination: dict[str, np.ndarray],
    texture: dict[str, np.ndarray],
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine threshold-relative color evidence and local regions by weighted score.

    Warm-bright appearance is used as a user-weighted ambiguity/score penalty cue,
    never as inferred solar position. Texture is supporting evidence only.
    """
    weights = {
        "argd": float(params.get("hybrid_w_argd", 0.35)),
        "nsv": float(params.get("hybrid_w_nsv", 0.25)),
        "ratio": float(params.get("hybrid_w_ratio", 0.20)),
        "local": float(params.get("hybrid_w_local", 0.20)),
    }
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Hybrid weights must be non-negative with positive total.")
    cutoff = float(params.get("hybrid_threshold", 0.55))
    ambiguity_width = float(params.get("ambiguity_width", 0.12))
    if not 0 <= cutoff <= 1 or ambiguity_width < 0:
        raise ValueError("Hybrid threshold must be in [0, 1] and ambiguity width non-negative.")
    scores = {
        "argd": _cloud_score(argd, float(params.get("argd_threshold", 0.0)), "higher", 0.08),
        "nsv": _cloud_score(nsv, float(params.get("nsv_threshold", -0.3)), "lower", 0.08),
        "ratio": _cloud_score(
            normalized_ratio, float(params.get("ratio_threshold", 0.0)), "lower", 0.08
        ),
        "local": (local_mask > 0).astype(np.float32),
    }
    total_weight = sum(weights.values())
    score = sum(scores[name] * weights[name] for name in weights) / total_weight

    texture_support = np.clip(
        texture["local_std"] * float(params.get("texture_weight", 0.0)), 0.0, 0.25
    )
    score = np.clip(score + texture_support * (score - 0.5), 0.0, 1.0)
    mode = str(params.get("sunlight_mode", "Warm-bright + user region"))
    if mode == "Warm-bright cue":
        interference = illumination["warm_bright_candidates"].astype(np.float32)
    elif mode == "User region":
        interference = illumination["user_sunlight_region"].astype(np.float32)
    elif mode == "Warm-bright + user region":
        interference = np.maximum(
            illumination["warm_bright_candidates"].astype(np.float32),
            illumination["user_sunlight_region"].astype(np.float32),
        )
    else:
        interference = np.zeros_like(score, dtype=np.float32)
    penalty = float(params.get("sunlight_penalty", 0.35))
    score *= 1.0 - np.clip(penalty, 0.0, 1.0) * interference

    binary = score >= cutoff
    ambiguous = (score >= cutoff - ambiguity_width) & (score <= cutoff + ambiguity_width)
    return binary.astype(np.uint8) * 255, ambiguous.astype(np.uint8) * 255, score.astype(np.float32)
