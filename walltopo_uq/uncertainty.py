"""Uncertainty aggregation and selective-reporting utilities."""

from __future__ import annotations

import numpy as np


def pixel_vacuity(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha)
    return 2.0 / np.clip(alpha.sum(axis=1), 1.0e-6, None)


def _binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def _topology_signature(mask: np.ndarray) -> tuple[float, float, float]:
    import cv2

    from .targets import node_maps, skeleton_from_mask

    binary = np.asarray(mask) > 0
    skeleton = skeleton_from_mask(binary)
    endpoints, junctions = node_maps(skeleton)
    component_count, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return (
        float(max(0, component_count - 1)),
        float(endpoints.sum()),
        float(junctions.sum()),
    )


def topology_instability(
    predictions: list[np.ndarray],
    eps: float = 1.0e-6,
) -> float:
    """Measure graph-level variation over geometrically valid TTA views.

    The score combines pairwise centerline disagreement with changes in the
    connected-component, endpoint, and junction counts. It is zero when all
    views produce the same graph and increases as the reconstructed crack graph
    becomes view dependent.
    """

    if len(predictions) < 2:
        return 0.0
    binary = [np.asarray(prediction) > 0 for prediction in predictions]
    overlaps = [
        1.0 - _binary_dice(binary[left], binary[right])
        for left in range(len(binary))
        for right in range(left + 1, len(binary))
    ]
    signatures = np.asarray([_topology_signature(prediction) for prediction in binary])
    reference = np.maximum(np.mean(signatures, axis=0), 1.0)
    signature_change = np.mean(np.abs(signatures - signatures.mean(axis=0)) / reference)
    topology_change = 0.65 * float(np.mean(overlaps)) + 0.35 * float(signature_change)
    return float(np.clip(topology_change / (1.0 + eps), 0.0, 1.0))


def geometry_dispersion(
    width_values: list[float],
    length_values: list[float],
) -> float:
    """Coefficient of variation of geometry descriptors under TTA."""

    terms: list[float] = []
    for values in (width_values, length_values):
        array = np.asarray(values, dtype=np.float64)
        if array.size < 2:
            terms.append(0.0)
            continue
        mean = abs(float(array.mean()))
        if mean <= 1.0e-6:
            terms.append(float(np.std(array) > 1.0e-6))
        else:
            terms.append(float(np.std(array) / mean))
    return float(np.clip(np.mean(terms), 0.0, 1.0))


def orientation_dispersion(
    orientation_maps: list[np.ndarray],
    support_mask: np.ndarray,
) -> float:
    """Mean sign-invariant angular dispersion of orientation vectors."""

    support = np.asarray(support_mask) > 0
    if len(orientation_maps) < 2 or not support.any():
        return 0.0
    stacked = np.stack(
        [np.asarray(item, dtype=np.float64)[:, support] for item in orientation_maps],
        axis=0,
    )
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    stacked = stacked / np.clip(norms, 1.0e-6, None)
    reference = stacked[0][None, ...]
    cosine = np.abs((stacked * reference).sum(axis=1)).clip(0.0, 1.0)
    return float(np.clip(np.degrees(np.arccos(cosine)).mean() / 90.0, 0.0, 1.0))


def normalized_mahalanobis(
    feature: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    scale: float,
) -> float:
    """Robust diagonal Mahalanobis score normalized by a calibration quantile."""

    feature = np.asarray(feature, dtype=np.float64)
    difference = feature - np.asarray(mean, dtype=np.float64)
    deviation = np.asarray(standard_deviation, dtype=np.float64)
    distance = float(np.sqrt(np.mean((difference / np.clip(deviation, 1.0e-6, None)) ** 2)))
    return float(np.clip(distance / max(float(scale), 1.0e-6), 0.0, 1.0))


def report_uncertainty(
    pixel_vacuity_map: np.ndarray,
    crack_mask: np.ndarray,
    topology_instability_value: float,
    geometry_dispersion: float,
    domain_shift_score: float = 0.0,
    weights: tuple[float, float, float, float] = (0.45, 0.25, 0.20, 0.10),
) -> float:
    mask = np.asarray(crack_mask) > 0
    if mask.any():
        pixel_term = float(np.asarray(pixel_vacuity_map)[mask].mean())
    else:
        pixel_term = float(np.asarray(pixel_vacuity_map).mean())
    components = np.asarray(
        [
            pixel_term,
            topology_instability_value,
            geometry_dispersion,
            domain_shift_score,
        ],
        dtype=np.float64,
    )
    weights_array = np.asarray(weights, dtype=np.float64)
    weights_array = weights_array / max(weights_array.sum(), 1.0e-6)
    return float(np.clip((components * weights_array).sum(), 0.0, 1.0))


def error_detection_auroc(errors: np.ndarray, uncertainty: np.ndarray) -> float:
    """AUROC for ranking erroneous images above correct images."""

    errors = np.asarray(errors, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    positive = errors > np.median(errors)
    negative = ~positive
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    if positive_count == 0 or negative_count == 0:
        return 0.5
    order = np.argsort(uncertainty)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, uncertainty.size + 1, dtype=np.float64)
    positive_rank_sum = float(ranks[positive].sum())
    statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(statistic / (positive_count * negative_count))


def risk_coverage_curve(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    coverages: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    errors = np.asarray(errors, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if errors.shape != uncertainty.shape:
        raise ValueError("errors and uncertainty must have the same shape")
    order = np.argsort(uncertainty)
    sorted_errors = errors[order]
    if coverages is None:
        coverages = np.linspace(0.05, 1.0, 20)
    risks = []
    for coverage in coverages:
        count = max(1, int(np.ceil(coverage * errors.size)))
        risks.append(float(sorted_errors[:count].mean()))
    return np.asarray(coverages), np.asarray(risks)


def area_under_risk_coverage(errors: np.ndarray, uncertainty: np.ndarray) -> float:
    coverages, risks = risk_coverage_curve(errors, uncertainty)
    return float(np.trapezoid(risks, coverages))


def selective_metrics(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    secondary_error: np.ndarray | None = None,
    coverage: float = 0.8,
) -> dict[str, float]:
    """Risk/performance after retaining the least uncertain samples."""

    errors = np.asarray(errors, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if errors.size == 0:
        return {
            "aurc": 0.0,
            "error_detection_auroc": 0.5,
            "selective_coverage": 0.0,
            "selective_error": 0.0,
            "selective_secondary_error": 0.0,
        }
    order = np.argsort(uncertainty)
    count = min(errors.size, max(1, int(np.ceil(coverage * errors.size))))
    accepted = order[:count]
    secondary = errors if secondary_error is None else np.asarray(secondary_error)
    return {
        "aurc": area_under_risk_coverage(errors, uncertainty),
        "error_detection_auroc": error_detection_auroc(errors, uncertainty),
        "selective_coverage": float(count / errors.size),
        "selective_error": float(errors[accepted].mean()),
        "selective_secondary_error": float(secondary[accepted].mean()),
    }
