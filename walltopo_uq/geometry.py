"""Geometry extraction from predicted masks and centerlines."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import convolve


def _as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[0]
    return array


def skeleton_length(skeleton: np.ndarray) -> float:
    skeleton = (_as_2d(skeleton) > 0).astype(np.float32)
    if not skeleton.any():
        return 0.0
    neighbours = convolve(
        skeleton,
        np.ones((3, 3), dtype=np.float32),
        mode="constant",
        cval=0.0,
    ) - skeleton
    straight = ((neighbours == 1) | (neighbours == 2)).sum()
    diagonal = (neighbours >= 3).sum()
    return float(straight + np.sqrt(2.0) * diagonal * 0.5)


def endpoint_count(skeleton: np.ndarray) -> int:
    skeleton = (_as_2d(skeleton) > 0).astype(np.uint8)
    neighbours = convolve(
        skeleton.astype(np.float32),
        np.ones((3, 3), dtype=np.float32),
        mode="constant",
        cval=0.0,
    ) - skeleton
    return int(np.count_nonzero((skeleton > 0) & (neighbours == 1)))


def junction_count(skeleton: np.ndarray) -> int:
    skeleton = (_as_2d(skeleton) > 0).astype(np.uint8)
    neighbours = convolve(
        skeleton.astype(np.float32),
        np.ones((3, 3), dtype=np.float32),
        mode="constant",
        cval=0.0,
    ) - skeleton
    return int(np.count_nonzero((skeleton > 0) & (neighbours >= 3)))


def component_count(mask: np.ndarray) -> int:
    binary = (_as_2d(mask) > 0).astype(np.uint8)
    count, _ = cv2.connectedComponents(binary, connectivity=8)
    return max(0, int(count) - 1)


def extract_geometry(
    mask: np.ndarray,
    skeleton: np.ndarray,
    width: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
    pixels_per_unit: float | None = None,
) -> dict[str, float]:
    mask = (_as_2d(mask) > 0).astype(np.float32)
    skeleton = (_as_2d(skeleton) > 0).astype(np.float32)
    height, width_size = mask.shape
    scale = 1.0 if pixels_per_unit is None else float(pixels_per_unit)
    length = skeleton_length(skeleton) * scale
    mask_area = float(mask.sum()) * scale * scale
    wall_area = float(height * width_size) * scale * scale
    components = component_count(mask)
    endpoints = endpoint_count(skeleton)
    junctions = junction_count(skeleton)

    width_values = np.zeros(0, dtype=np.float32)
    if width is not None and skeleton.any():
        width_values = np.asarray(width)[0] if np.asarray(width).ndim == 3 else np.asarray(width)
        width_values = width_values[skeleton > 0].astype(np.float32)
        width_values = width_values * max(height, width_size) * scale
    orientation_concentration = 0.0
    if orientation is not None and skeleton.any():
        orientation = np.asarray(orientation)
        if orientation.ndim == 3 and orientation.shape[0] >= 2:
            vectors = orientation[:2, skeleton > 0]
            norm = np.linalg.norm(vectors, axis=0, keepdims=True)
            unit = vectors / np.clip(norm, 1.0e-6, None)
            concentration = np.abs(unit).mean(axis=1)
            orientation_concentration = float(np.mean(concentration))

    return {
        "length": float(length),
        "length_density": float(length / max(wall_area, 1.0)),
        "area": mask_area,
        "area_ratio": float(mask_area / max(wall_area, 1.0)),
        "width_mean": float(width_values.mean()) if width_values.size else 0.0,
        "width_p90": float(np.quantile(width_values, 0.9)) if width_values.size else 0.0,
        "width_max": float(width_values.max()) if width_values.size else 0.0,
        "components": float(components),
        "endpoints": float(endpoints),
        "junctions": float(junctions),
        "branch_density": float(junctions / max(length, 1.0)),
        "orientation_concentration": orientation_concentration,
    }


def visual_screening_score(
    geometry: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Rule-based visual priority score from normalized descriptors."""

    weights = weights or {
        "length_density": 0.25,
        "width_p90": 0.30,
        "branch_density": 0.20,
        "components": 0.10,
        "orientation_concentration": 0.15,
    }
    length_term = np.clip(geometry["length_density"] / 0.08, 0.0, 1.0)
    width_term = np.clip(geometry["width_p90"] / 12.0, 0.0, 1.0)
    branch_term = np.clip(geometry["branch_density"] / 0.25, 0.0, 1.0)
    component_term = np.clip(geometry["components"] / 5.0, 0.0, 1.0)
    orientation_term = np.clip(geometry["orientation_concentration"], 0.0, 1.0)
    return float(
        weights["length_density"] * length_term
        + weights["width_p90"] * width_term
        + weights["branch_density"] * branch_term
        + weights["components"] * component_term
        + weights["orientation_concentration"] * orientation_term
    )
