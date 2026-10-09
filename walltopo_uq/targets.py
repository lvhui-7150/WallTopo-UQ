"""Derived supervision targets used by WallTopo-UQ."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter

try:
    from skimage.morphology import skeletonize
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise RuntimeError("scikit-image is required for skeletonization") from exc


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    binary = _binary_mask(mask)
    if not binary.any():
        return np.zeros_like(binary, dtype=np.float32)
    return skeletonize(binary).astype(np.float32)


def node_maps(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skeleton = (np.asarray(skeleton) > 0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0.0
    neighbours = convolve(skeleton, kernel, mode="constant", cval=0.0)
    endpoints = ((skeleton > 0) & (neighbours == 1)).astype(np.float32)
    junctions = ((skeleton > 0) & (neighbours >= 3)).astype(np.float32)
    return endpoints, junctions


def orientation_from_skeleton(skeleton: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    skeleton = np.asarray(skeleton, dtype=np.float32)
    if not skeleton.any():
        return np.zeros((2, *skeleton.shape), dtype=np.float32)
    smoothed = gaussian_filter(skeleton, sigma=sigma, mode="reflect")
    gradient_y, gradient_x = np.gradient(smoothed)
    jxx = gaussian_filter(gradient_x * gradient_x, sigma=sigma, mode="reflect")
    jxy = gaussian_filter(gradient_x * gradient_y, sigma=sigma, mode="reflect")
    jyy = gaussian_filter(gradient_y * gradient_y, sigma=sigma, mode="reflect")
    gradient_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    tangent_angle = gradient_angle + 0.5 * np.pi
    orientation = np.stack(
        [np.cos(tangent_angle), np.sin(tangent_angle)],
        axis=0,
    ).astype(np.float32)
    orientation[:, skeleton <= 0] = 0.0
    return orientation


def width_from_mask(mask: np.ndarray, normalize: bool = True) -> np.ndarray:
    binary = _binary_mask(mask)
    width = 2.0 * distance_transform_edt(binary).astype(np.float32)
    if normalize:
        width = width / max(binary.shape)
    return width


def derive_targets(mask: np.ndarray, normalize_width: bool = True) -> dict[str, np.ndarray]:
    binary = _binary_mask(mask).astype(np.float32)
    skeleton = skeleton_from_mask(binary)
    endpoints, junctions = node_maps(skeleton)
    return {
        "mask": binary,
        "skeleton": skeleton,
        "endpoint": endpoints,
        "junction": junctions,
        "orientation": orientation_from_skeleton(skeleton),
        "width": width_from_mask(binary, normalize=normalize_width),
    }


def geometry_severity(
    mask: np.ndarray,
    skeleton: np.ndarray | None = None,
    num_classes: int = 5,
) -> int:
    """Synthetic-only severity proxy for smoke tests.

    Publication experiments must use an independent expert rubric. This helper
    exists only to exercise the ordinal branch when manual labels are absent.
    """

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    binary = _binary_mask(mask)
    skeleton = skeleton_from_mask(binary) if skeleton is None else (skeleton > 0)
    if not binary.any():
        return 0
    _, junctions = node_maps(skeleton)
    length_density = float(skeleton.sum()) / float(binary.size)
    width = float(width_from_mask(binary, normalize=True).max())
    branch_density = float(junctions.sum()) / max(float(skeleton.sum()), 1.0)
    score = 1.4 * length_density + 8.0 * width + 0.6 * branch_density
    thresholds = np.linspace(0.02, 0.22, num_classes - 1)
    return int(np.digitize(score, thresholds))
