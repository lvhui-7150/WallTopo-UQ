"""Topology, geometry, ordinal, and evidential objectives."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _binary_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * intersection + eps) / (denominator + eps)


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=dims)
    false_positive = (probability * (1.0 - target)).sum(dim=dims)
    false_negative = ((1.0 - probability) * target).sum(dim=dims)
    tversky = (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps
    )
    return (1.0 - tversky).pow(gamma).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = 1.0 - _binary_dice(logits, target).mean()
    tversky = focal_tversky_loss(logits, target)
    return bce + dice + tversky


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeleton(x: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    skeleton = F.relu(x - soft_open(x))
    for _ in range(iterations - 1):
        x = soft_erode(x)
        skeleton = skeleton + F.relu(x - soft_open(x))
    return skeleton


def soft_cldice(
    logits: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 3,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    prediction_skeleton = soft_skeleton(probability, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    dims = tuple(range(1, probability.ndim))
    topology_precision = (
        (prediction_skeleton * target).sum(dim=dims) + eps
    ) / (prediction_skeleton.sum(dim=dims) + eps)
    topology_sensitivity = (
        (target_skeleton * probability).sum(dim=dims) + eps
    ) / (target_skeleton.sum(dim=dims) + eps)
    return (2.0 * topology_precision * topology_sensitivity) / (
        topology_precision + topology_sensitivity + eps
    )


def node_loss(
    endpoint_logits: torch.Tensor,
    junction_logits: torch.Tensor,
    endpoint_target: torch.Tensor,
    junction_target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    def focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
        return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()

    return focal(endpoint_logits, endpoint_target) + focal(junction_logits, junction_target)


def topology_loss(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    cl_dice_weight: float,
    node_weight: float,
) -> torch.Tensor:
    skeleton_dice = 1.0 - _binary_dice(
        outputs["skeleton_logits"],
        target["skeleton"],
    ).mean()
    topology_dice = 1.0 - soft_cldice(
        outputs["mask_logits"],
        target["mask"],
    ).mean()
    nodes = node_loss(
        outputs["endpoint_logits"],
        outputs["junction_logits"],
        target["endpoint"],
        target["junction"],
    )
    return skeleton_dice + cl_dice_weight * topology_dice + node_weight * nodes


def geometry_loss(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    width_weight: float,
    width_consistency_weight: float,
    orientation_weight: float,
) -> torch.Tensor:
    centerline = target["skeleton"]
    band = F.max_pool2d(centerline, kernel_size=5, stride=1, padding=2)
    band = F.max_pool2d(band, kernel_size=5, stride=1, padding=2)
    band_area = band.sum().clamp_min(1.0)

    width_error = F.smooth_l1_loss(
        outputs["width"] * band,
        target["width"] * band,
        reduction="sum",
    ) / band_area
    orientation_target = target["orientation"]
    orientation_dot = torch.abs(
        (outputs["orientation"] * orientation_target).sum(dim=1, keepdim=True)
    )
    orientation_error = ((1.0 - orientation_dot) * centerline).sum() / centerline.sum().clamp_min(1.0)

    soft_distance = differentiable_distance(target["mask"], steps=16)
    predicted_width_from_mask = 2.0 * soft_distance
    consistency = F.smooth_l1_loss(
        outputs["width"] * band,
        predicted_width_from_mask * band,
        reduction="sum",
    ) / band_area
    return (
        width_weight * width_error
        + width_consistency_weight * consistency
        + orientation_weight * orientation_error
    )


def differentiable_distance(mask: torch.Tensor, steps: int = 16) -> torch.Tensor:
    """Differentiable chessboard-distance approximation."""

    active = mask.clamp(0.0, 1.0)
    distance = torch.zeros_like(active)
    for step in range(1, steps + 1):
        dilated = F.max_pool2d(active, kernel_size=3, stride=1, padding=1)
        newly_active = (dilated - active).clamp_min(0.0)
        distance = distance + step * newly_active
        active = dilated
    return distance / max(mask.shape[-1], mask.shape[-2])


def ordinal_loss(
    severity_logits: torch.Tensor,
    severity: torch.Tensor,
    severity_valid: torch.Tensor,
) -> torch.Tensor:
    valid = severity_valid > 0.5
    if not valid.any():
        return severity_logits.sum() * 0.0
    logits = severity_logits[valid]
    labels = severity[valid]
    thresholds = torch.arange(logits.shape[1], device=logits.device)[None, :]
    targets = (labels[:, None] > thresholds).float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def evidential_loss(
    evidence: torch.Tensor,
    target: torch.Tensor,
    kl_weight: float = 0.05,
) -> torch.Tensor:
    alpha = evidence.clamp_min(1.0 + 1.0e-6)
    strength = alpha.sum(dim=1, keepdim=True)
    target_1 = target.clamp(0.0, 1.0)
    target_0 = 1.0 - target_1
    expected_nll = (
        target_1 * (torch.digamma(strength) - torch.digamma(alpha[:, 1:2]))
        + target_0 * (torch.digamma(strength) - torch.digamma(alpha[:, 0:1]))
    ).mean()

    observed = torch.cat([target_0, target_1], dim=1)
    tilde_alpha = observed + (1.0 - observed) * alpha
    tilde_strength = tilde_alpha.sum(dim=1, keepdim=True)
    log_beta = torch.lgamma(tilde_alpha).sum(dim=1, keepdim=True) - torch.lgamma(tilde_strength)
    kl = (
        log_beta
        + (tilde_strength - tilde_alpha.sum(dim=1, keepdim=True))
        * torch.digamma(tilde_strength)
        + ((tilde_alpha - 1.0) * (torch.digamma(tilde_alpha) - torch.digamma(tilde_strength))).sum(
            dim=1,
            keepdim=True,
        )
    ).mean()
    return expected_nll + kl_weight * kl


class WallTopoLoss(nn.Module):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = dict(weights)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        epoch_fraction: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        seg = segmentation_loss(outputs["mask_logits"], batch["mask"])
        topo = topology_loss(
            outputs,
            batch,
            cl_dice_weight=float(self.weights.get("cl_dice", 1.0)),
            node_weight=float(self.weights.get("node", 1.0)),
        )
        geometry = geometry_loss(
            outputs,
            batch,
            width_weight=float(self.weights.get("width", 1.0)),
            width_consistency_weight=float(self.weights.get("width_consistency", 0.25)),
            orientation_weight=float(self.weights.get("orientation", 0.25)),
        )
        ordinal = ordinal_loss(
            outputs["severity_logits"],
            batch["severity"],
            batch["severity_valid"],
        )
        evidence_kl_weight = float(self.weights.get("evidence_kl", 0.05))
        evidence_kl_weight *= min(1.0, max(0.0, epoch_fraction))
        evidence = evidential_loss(outputs["evidence"], batch["mask"], evidence_kl_weight)

        total = (
            float(self.weights.get("seg", 1.0)) * seg
            + float(self.weights.get("topo", 0.7)) * topo
            + float(self.weights.get("geo", 0.7)) * geometry
            + float(self.weights.get("ordinal", 0.2)) * ordinal
            + float(self.weights.get("evidence", 0.25)) * evidence
        )
        components = {
            "loss": total.detach(),
            "loss_seg": seg.detach(),
            "loss_topo": topo.detach(),
            "loss_geo": geometry.detach(),
            "loss_ordinal": ordinal.detach(),
            "loss_evidence": evidence.detach(),
        }
        return total, components
