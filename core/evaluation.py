import cv2
import numpy as np

def prepare_ground_truth(mask_image, target_shape):
    """
    Convert an uploaded annotation to a binary mask without changing dimensions.

    The uploaded image must exactly match the ROI dimensions. White (>=127)
    represents cloud; darker values represent non-cloud. Silent resizing is
    intentionally not performed because it can invalidate alignment.
    """
    if mask_image is None:
        raise ValueError("Ground-truth mask is missing.")
    if mask_image.ndim == 3:
        if mask_image.shape[2] != 3:
            raise ValueError("Ground-truth mask must be a grayscale or RGB image.")
        gray = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
    elif mask_image.ndim == 2:
        gray = mask_image
    else:
        raise ValueError("Ground-truth mask must be a 2D or 3D image.")

    if tuple(gray.shape[:2]) != tuple(target_shape[:2]):
        raise ValueError(
            f"Ground-truth dimensions {gray.shape[1]}x{gray.shape[0]} do not match "
            f"the selected ROI {target_shape[1]}x{target_shape[0]}."
        )
    if not np.isin(np.unique(gray), (0, 1, 255)).all():
        raise ValueError(
            "Ground-truth mask must be binary: use black (0) for non-cloud and "
            "white (255) for cloud."
        )
    return (gray > 0).astype(np.uint8)

def calculate_metrics(predicted_mask, ground_truth_mask, valid_mask=None):
    pred = np.asarray(predicted_mask)
    truth = np.asarray(ground_truth_mask)
    if pred.ndim != 2 or truth.ndim != 2:
        raise ValueError("Predicted and ground-truth masks must be 2-dimensional.")
    if pred.shape != truth.shape:
        raise ValueError(
            f"Mask dimensions differ: predicted {pred.shape[1]}x{pred.shape[0]}, "
            f"ground truth {truth.shape[1]}x{truth.shape[0]}."
        )
    valid = np.ones(pred.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != pred.shape:
        raise ValueError("Valid-sky mask dimensions must match prediction and ground truth.")
    if not valid.any():
        raise ValueError("No valid pixels are available for evaluation.")
    pred = (pred > 0) & valid
    truth = (truth > 0) & valid
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    tn = int(np.logical_and.reduce((np.logical_not(pred), np.logical_not(truth), valid)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision * 100,
        "recall": recall * 100,
        "iou": iou * 100,
        "f1": f1 * 100,
        "accuracy": accuracy * 100,
    }


def evaluate_methods(method_results, ground_truth_mask, valid_mask=None):
    """Evaluate every available refined method mask against one aligned reference."""
    metrics_by_method = {}
    for name, method in method_results.items():
        if not isinstance(method, dict):
            continue
        prediction = method.get("refined", method.get("mask"))
        if prediction is None:
            continue
        metrics_by_method[name] = calculate_metrics(
            prediction,
            ground_truth_mask,
            valid_mask=valid_mask,
        )
    if not metrics_by_method:
        raise ValueError("No segmentation methods provide masks for evaluation.")
    return metrics_by_method


def make_comparison_map(predicted_mask, ground_truth_mask, valid_mask=None):
    """
    RGB comparison:
    green = true positive, red = false positive, blue = false negative.
    """
    if np.asarray(predicted_mask).shape != np.asarray(ground_truth_mask).shape:
        raise ValueError("Predicted and ground-truth mask dimensions must match.")
    pred = predicted_mask > 0
    truth = ground_truth_mask > 0
    valid = np.ones(pred.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != pred.shape:
        raise ValueError("Valid-sky mask dimensions must match prediction and ground truth.")
    result = np.zeros((*pred.shape, 3), dtype=np.uint8)
    result[np.logical_and.reduce((pred, truth, valid))] = (46, 160, 90)
    result[np.logical_and.reduce((pred, np.logical_not(truth), valid))] = (225, 75, 75)
    result[np.logical_and.reduce((np.logical_not(pred), truth, valid))] = (65, 135, 220)
    result[~valid] = (120, 120, 120)
    return result
