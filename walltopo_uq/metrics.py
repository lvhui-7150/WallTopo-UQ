"""Segmentation, topology, geometry, severity, and calibration metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .geometry import component_count, endpoint_count, junction_count, skeleton_length
from .targets import node_maps, skeleton_from_mask


EPS = 1.0e-7


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def binary_scores(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    true_positive = float(np.logical_and(prediction, target).sum())
    false_positive = float(np.logical_and(prediction, ~target).sum())
    false_negative = float(np.logical_and(~prediction, target).sum())
    precision = true_positive / max(true_positive + false_positive, EPS)
    recall = true_positive / max(true_positive + false_negative, EPS)
    dice = 2.0 * true_positive / max(2.0 * true_positive + false_positive + false_negative, EPS)
    iou = true_positive / max(true_positive + false_positive + false_negative, EPS)
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }


def boundary_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance_ratio: float = 0.01,
) -> float:
    prediction = prediction.astype(np.uint8)
    target = target.astype(np.uint8)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    prediction_boundary = prediction - cv2.erode(prediction, kernel)
    target_boundary = target - cv2.erode(target, kernel)
    tolerance = max(1, int(round(tolerance_ratio * max(target.shape))))
    kernel_tolerance = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    prediction_dilated = cv2.dilate(prediction_boundary, kernel_tolerance)
    target_dilated = cv2.dilate(target_boundary, kernel_tolerance)
    precision = np.logical_and(prediction_boundary > 0, target_dilated > 0).sum() / max(
        prediction_boundary.sum(),
        1,
    )
    recall = np.logical_and(target_boundary > 0, prediction_dilated > 0).sum() / max(
        target_boundary.sum(),
        1,
    )
    return float(2.0 * precision * recall / max(precision + recall, EPS))


def hard_cldice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_skeleton = skeleton_from_mask(prediction)
    target_skeleton = skeleton_from_mask(target)
    if not prediction_skeleton.any() and not target_skeleton.any():
        return 1.0
    topology_precision = float((prediction_skeleton * target).sum()) / max(
        float(prediction_skeleton.sum()),
        EPS,
    )
    topology_sensitivity = float((target_skeleton * prediction).sum()) / max(
        float(target_skeleton.sum()),
        EPS,
    )
    return float(
        2.0
        * topology_precision
        * topology_sensitivity
        / max(topology_precision + topology_sensitivity, EPS)
    )


def node_localization_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: int = 2,
) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    kernel = np.ones(
        (2 * max(0, int(tolerance)) + 1, 2 * max(0, int(tolerance)) + 1),
        dtype=np.uint8,
    )
    prediction_dilated = cv2.dilate(prediction.astype(np.uint8), kernel) > 0
    target_dilated = cv2.dilate(target.astype(np.uint8), kernel) > 0
    precision = np.logical_and(prediction, target_dilated).sum() / max(
        prediction.sum(),
        1,
    )
    recall = np.logical_and(target, prediction_dilated).sum() / max(
        target.sum(),
        1,
    )
    return float(2.0 * precision * recall / max(precision + recall, EPS))


def hole_count(mask: np.ndarray) -> int:
    binary = np.asarray(mask) > 0
    inverse = (~binary).astype(np.uint8)
    count, labels = cv2.connectedComponents(inverse, connectivity=4)
    if count == 0:
        return 0
    border_labels = set(labels[0, :]) | set(labels[-1, :])
    border_labels |= set(labels[:, 0]) | set(labels[:, -1])
    border_labels.discard(0)
    return max(0, int(count - 1 - len(border_labels)))


def topology_scores(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    prediction_skeleton = skeleton_from_mask(prediction)
    target_skeleton = skeleton_from_mask(target)
    skeleton_scores = binary_scores(prediction_skeleton > 0, target_skeleton > 0)
    target_components = component_count(target)
    prediction_components = component_count(prediction)
    target_endpoints = endpoint_count(target_skeleton)
    prediction_endpoints = endpoint_count(prediction_skeleton)
    target_junctions = junction_count(target_skeleton)
    prediction_junctions = junction_count(prediction_skeleton)
    target_endpoint_map, target_junction_map = node_maps(target_skeleton)
    prediction_endpoint_map, prediction_junction_map = node_maps(prediction_skeleton)
    break_error = max(0, prediction_components - target_components) / max(
        target_components,
        1,
    )
    merge_error = max(0, target_components - prediction_components) / max(
        target_components,
        1,
    )
    endpoint_error = abs(prediction_endpoints - target_endpoints) / max(target_endpoints, 1)
    junction_error = abs(prediction_junctions - target_junctions) / max(target_junctions, 1)
    prediction_length = skeleton_length(prediction_skeleton)
    target_length = skeleton_length(target_skeleton)
    target_holes = hole_count(target)
    prediction_holes = hole_count(prediction)
    return {
        "cl_dice": hard_cldice(prediction, target),
        "skeleton_dice": skeleton_scores["dice"],
        "skeleton_iou": skeleton_scores["iou"],
        "component_error": float(
            abs(prediction_components - target_components) / max(target_components, 1)
        ),
        "false_merge_error": float(merge_error),
        "break_rate": float(break_error),
        "connected_component_error": float(
            abs(prediction_components - target_components) / max(target_components, 1)
        ),
        "betti_0_error": float(
            abs(prediction_components - target_components)
        ),
        "betti_1_error": float(abs(prediction_holes - target_holes)),
        "endpoint_f1": node_localization_f1(
            prediction_endpoint_map,
            target_endpoint_map,
        ),
        "junction_f1": node_localization_f1(
            prediction_junction_map,
            target_junction_map,
        ),
        "endpoint_error": float(endpoint_error),
        "junction_error": float(junction_error),
        "length_mae_px": float(abs(prediction_length - target_length)),
        "length_mape": float(
            abs(prediction_length - target_length) / max(target_length, 1.0)
        ),
    }


def geometry_scores(
    prediction_width: np.ndarray,
    target_width: np.ndarray,
    target_skeleton: np.ndarray,
) -> dict[str, float]:
    skeleton = target_skeleton > 0
    if not skeleton.any():
        return {"width_mae_px": 0.0, "width_mape": 0.0}
    prediction = prediction_width[skeleton]
    target = target_width[skeleton]
    absolute_error = np.abs(prediction - target)
    return {
        "width_mae_px": float(absolute_error.mean()),
        "width_mape": float(
            (absolute_error / np.clip(target, 0.5, None)).mean()
        ),
    }


def orientation_scores(
    prediction_orientation: np.ndarray,
    target_orientation: np.ndarray,
    target_skeleton: np.ndarray,
) -> dict[str, float]:
    skeleton = target_skeleton > 0
    if not skeleton.any():
        return {"orientation_mae_deg": 0.0}
    prediction = prediction_orientation[:, skeleton]
    target = target_orientation[:, skeleton]
    prediction = prediction / np.clip(
        np.linalg.norm(prediction, axis=0, keepdims=True),
        EPS,
        None,
    )
    target = target / np.clip(
        np.linalg.norm(target, axis=0, keepdims=True),
        EPS,
        None,
    )
    cosine = np.abs((prediction * target).sum(axis=0)).clip(0.0, 1.0)
    return {"orientation_mae_deg": float(np.degrees(np.arccos(cosine)).mean())}


def expected_calibration_error(
    probability: np.ndarray,
    target: np.ndarray,
    bins: int = 15,
) -> float:
    probability = probability.reshape(-1)
    target = target.reshape(-1).astype(np.float32)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = (probability >= low) & (probability < high)
        if not mask.any():
            continue
        confidence = probability[mask].mean()
        accuracy = target[mask].mean()
        error += float(mask.mean()) * abs(confidence - accuracy)
    return float(error)


def probability_scores(
    probability: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    probability = np.clip(probability.reshape(-1), EPS, 1.0 - EPS)
    target = target.reshape(-1).astype(np.float32)
    return {
        "brier": float(np.mean((probability - target) ** 2)),
        "nll": float(-np.mean(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))),
        "ece": expected_calibration_error(probability, target),
    }


def quadratic_weighted_kappa(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> float:
    prediction = prediction.astype(np.int64)
    target = target.astype(np.int64)
    if prediction.size == 0:
        return 0.0
    observed = np.zeros((num_classes, num_classes), dtype=np.float64)
    for truth, estimate in zip(target, prediction):
        if 0 <= truth < num_classes and 0 <= estimate < num_classes:
            observed[truth, estimate] += 1.0
    if observed.sum() == 0:
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.zeros_like(observed)
    for row in range(num_classes):
        for column in range(num_classes):
            weights[row, column] = ((row - column) ** 2) / max((num_classes - 1) ** 2, 1)
    denominator = (weights * expected).sum()
    if denominator <= EPS:
        return 1.0
    return float(1.0 - (weights * observed).sum() / denominator)


def severity_scores(
    logits: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    selected = valid > 0.5
    if not selected.any():
        return {"severity_accuracy": 0.0, "severity_qwk": 0.0, "severity_mae": 0.0}
    logits = logits[selected]
    target = target[selected]
    cumulative = 1.0 / (1.0 + np.exp(-logits))
    prediction = (cumulative > 0.5).sum(axis=1)
    accuracy = float((prediction == target).mean())
    mae = float(np.abs(prediction - target).mean())
    qwk = quadratic_weighted_kappa(prediction, target, num_classes)
    return {
        "severity_accuracy": accuracy,
        "severity_qwk": qwk,
        "severity_mae": mae,
    }


def composite_score(metrics: dict[str, float]) -> float:
    geometry_score = 1.0 / (1.0 + metrics.get("width_mae_px", 0.0) / 4.0)
    topology_score = metrics.get("cl_dice", 0.0)
    segmentation_score = metrics.get("dice", 0.0)
    reliability_score = 1.0 - metrics.get("ece", 0.0)
    return float(
        0.40 * segmentation_score
        + 0.30 * topology_score
        + 0.15 * geometry_score
        + 0.15 * reliability_score
    )


def validate_batch(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    threshold: float = 0.5,
    pixels_per_unit: float | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    accumulator: dict[str, list[float]] = defaultdict(list)
    per_image: list[dict[str, Any]] = []

    mask_probability = torch.sigmoid(outputs["mask_logits"])
    skeleton_probability = torch.sigmoid(outputs["skeleton_logits"])
    evidence = outputs["evidence"]
    vacuity = 2.0 / evidence.sum(dim=1, keepdim=True).clamp_min(EPS)

    batch_size = mask_probability.shape[0]
    for index in range(batch_size):
        prediction = (_tensor_to_numpy(mask_probability[index, 0]) >= threshold)
        target = _tensor_to_numpy(batch["mask"][index, 0]) >= 0.5
        scores = binary_scores(prediction, target)
        scores.update(topology_scores(prediction, target))
        scores["boundary_f1"] = boundary_f1(prediction, target)

        probability_np = _tensor_to_numpy(mask_probability[index, 0])
        target_np = _tensor_to_numpy(batch["mask"][index, 0])
        scores.update(probability_scores(probability_np, target_np))

        predicted_skeleton = _tensor_to_numpy(skeleton_probability[index, 0]) >= threshold
        target_skeleton = _tensor_to_numpy(batch["skeleton"][index, 0]) >= 0.5
        width_scale = float(pixels_per_unit or max(target_np.shape))
        scores.update(
            geometry_scores(
                _tensor_to_numpy(outputs["width"][index, 0]) * width_scale,
                _tensor_to_numpy(batch["width"][index, 0]) * width_scale,
                target_skeleton,
            )
        )
        scores.update(
            orientation_scores(
                _tensor_to_numpy(outputs["orientation"][index]),
                _tensor_to_numpy(batch["orientation"][index]),
                target_skeleton,
            )
        )
        scores["mean_vacuity"] = float(_tensor_to_numpy(vacuity[index, 0]).mean())
        scores["image_error"] = 1.0 - scores["dice"]
        for key, value in scores.items():
            accumulator[key].append(float(value))
        per_image.append(scores)

    metrics = {key: float(np.mean(values)) for key, values in accumulator.items()}
    severity_logits = _tensor_to_numpy(outputs["severity_logits"])
    severity_target = _tensor_to_numpy(batch["severity"])
    severity_valid = _tensor_to_numpy(batch["severity_valid"])
    metrics.update(
        severity_scores(
            severity_logits,
            severity_target,
            severity_valid,
            num_classes=outputs["severity_logits"].shape[1] + 1,
        )
    )
    metrics["composite"] = composite_score(metrics)
    return metrics, per_image
