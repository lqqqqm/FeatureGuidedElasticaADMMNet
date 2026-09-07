"""Supervision for signed RGB gradients and a fixed-scale soft edge map."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.operators import dot_mp, vector_norm
from ..utils.structure import edge_strength, structure_targets


def _per_sample_mean(value: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    weight = weight.expand_as(value)
    mass = weight.sum(dim=(1, 2, 3))
    mean = (value * weight).sum(dim=(1, 2, 3)) / mass.clamp_min(1e-8)
    return mean, (mass > 1e-8).to(value.dtype)


def _active_mean(value: Tensor, active: Tensor) -> Tensor:
    return (value * active).sum() / active.sum().clamp_min(1)


def _region_mean(value: Tensor, region: Tensor) -> Tensor:
    mean, active = _per_sample_mean(value, region)
    return _active_mean(mean, active)


def _balanced_region_mean(value: Tensor, region: Tensor, edge_gt: Tensor) -> Tensor:
    """Give GT edge and nonedge mass equal influence in each valid sample.

    Soft memberships depend only on fixed GT, never on predicted E or G.
    If either class is absent, the remaining class retains full weight.
    """
    edge_mean, edge_active = _per_sample_mean(value, region * edge_gt)
    flat_mean, flat_active = _per_sample_mean(value, region * (1 - edge_gt))
    count = edge_active + flat_active
    mean = (edge_mean * edge_active + flat_mean * flat_active) / count.clamp_min(1)
    return _active_mean(mean, (count > 0).to(value.dtype))


def _edge_region_loss(logits: Tensor, target: Tensor, region: Tensor) -> Tensor:
    bce = _region_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), region)
    edge = logits.sigmoid()
    dims = (1, 2, 3)
    overlap = (region * edge * target).sum(dim=dims)
    # Squared soft Dice has a zero optimum when a soft prediction equals GT.
    denominator = (region * (edge.square() + target.square())).sum(dim=dims)
    dice = 1 - (2 * overlap + 1e-6) / (denominator + 1e-6)
    active = (region.sum(dim=dims) > 0).to(edge.dtype)
    return bce + _active_mean(dice, active)


def structure_losses(
    structure: Mapping,
    gt: Tensor,
    known: Tensor,
    *,
    edge_scale: float = 0.1,
    orientation_threshold: float = 0.05,
    scale_weights: Optional[Sequence[float]] = None,
    known_weight: float = 0.1,
) -> Dict[str, Tensor]:
    """Return unweighted differentiable structure losses, evaluated in FP32.

    Gradient L1 is averaged over signed components, balanced by GT edge and
    nonedge membership, and normalized separately within holes and known
    regions for each sample/scale. Scale weights are normalized to sum one;
    the default full/half/quarter weights are proportional to 1, 1/2, 1/4.
    All other terms use the full-resolution heads. Known supervision is
    added separately with ``known_weight``. Empty regions contribute zero.
    Orientation uses signed cosine per RGB vector on GT magnitude above
    ``orientation_threshold``; a zero predicted vector has unit penalty.
    """
    gradient = structure["gradient"].float()
    pyramid = structure.get("gradient_pyramid", [gradient])
    if not pyramid:
        raise ValueError("gradient_pyramid must contain its full-resolution gradient")
    gradients = [value.float() for value in pyramid]
    if gradient.shape != gradients[0].shape or gradient.shape != (gt.shape[0], 6, *gt.shape[-2:]):
        raise ValueError("structure gradient and first pyramid entry must match full-resolution GT")
    weights = list(scale_weights) if scale_weights is not None else [0.5 ** i for i in range(len(gradients))]
    if len(weights) != len(gradients):
        raise ValueError("structure_scale_weights must match gradient_pyramid length")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("structure_scale_weights must be nonnegative with positive sum")
    total_weight = sum(weights)
    targets = structure_targets(gt.float(), [value.shape[-2:] for value in gradients], edge_scale)
    known = known.detach().float()
    l_gradient = gradient.sum() * 0
    for weight, prediction, target, edge_gt in zip(
        weights, gradients, targets["gradient_pyramid"], targets["edge_pyramid"]
    ):
        if prediction.shape != target.shape:
            raise ValueError("each predicted gradient must match its RGB target shape")
        # Area fractions retain small holes that nearest-neighbor resizing can lose.
        mask = known if known.shape[-2:] == prediction.shape[-2:] else F.interpolate(
            known, size=prediction.shape[-2:], mode="area"
        )
        error = (prediction - target).abs()
        loss = _balanced_region_mean(error, 1 - mask, edge_gt)
        loss = loss + known_weight * _balanced_region_mean(error, mask, edge_gt)
        l_gradient = l_gradient + weight / total_weight * loss

    target = targets["gradient"]
    edge_gt = targets["edge"]
    logits = structure["edge_logits"].float()
    if logits.shape != edge_gt.shape:
        raise ValueError("edge_logits must have shape [B,1,H,W] matching GT")
    hole = 1 - known
    l_edge = _edge_region_loss(logits, edge_gt, hole)
    l_edge = l_edge + known_weight * _edge_region_loss(logits, edge_gt, known)

    target_norm = vector_norm(target)
    predicted_norm = vector_norm(gradient)
    cosine = dot_mp(gradient, target) / (predicted_norm.clamp_min(1e-6) * target_norm.clamp_min(1e-6))
    orientation_error = 1 - cosine.clamp(-1, 1)
    strong = (target_norm > orientation_threshold).to(gradient.dtype)
    l_orientation = _region_mean(orientation_error, hole * strong)
    l_orientation = l_orientation + known_weight * _region_mean(orientation_error, known * strong)

    # Neither prediction is detached, so consistency trains both heads.
    consistency_error = (logits.sigmoid() - edge_strength(gradient, edge_scale)).abs()
    l_consistency = _region_mean(consistency_error, hole)
    l_consistency = l_consistency + known_weight * _region_mean(consistency_error, known)
    return {"structure_grad": l_gradient, "structure_edge": l_edge,
            "structure_orientation": l_orientation, "structure_consistency": l_consistency}
