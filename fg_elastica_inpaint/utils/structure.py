"""Shared, fixed-scale RGB structure targets for losses and evaluation."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.operators import grad, vector_norm


def edge_strength(gradient: Tensor, scale: float = 0.1) -> Tensor:
    """Return ``1 - exp(-mean_RGB(|gradient|) / scale)`` as [B,1,H,W].

    ``gradient`` has channels [dx_R,dx_G,dx_B,dy_R,dy_G,dy_B]. The
    Euclidean vector norm has an exact zero value and a finite zero
    subgradient at zero. There is no per-image maximum normalization.
    Half-precision inputs are evaluated in FP32 for stable supervision.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("edge scale must be finite and positive")
    if gradient.ndim != 4 or gradient.shape[1] != 6:
        raise ValueError("gradient must have shape [B,6,H,W] in dx/dy RGB order")
    if gradient.dtype in (torch.float16, torch.bfloat16):
        gradient = gradient.float()
    magnitude = vector_norm(gradient).mean(dim=1, keepdim=True)
    return -torch.expm1(-magnitude / scale)


def structure_targets(
    gt: Tensor, sizes: Sequence[Tuple[int, int]], edge_scale: float = 0.1
) -> Dict[str, Union[Tensor, list[Tensor]]]:
    """Build detached targets at each requested (height, width), in order.

    Return ``gradient_pyramid`` and ``edge_pyramid`` lists; ``gradient``
    and ``edge`` alias their first entries. Each gradient is taken *after*
    antialiased bilinear GT resizing, so it uses that scale's pixel units.
    The caller supplies the already augmented RGB GT. Centering before
    interpolation preserves exact zero gradients for constant images.
    """
    if gt.ndim != 4 or gt.shape[1] != 3:
        raise ValueError("gt must have shape [B,3,H,W]")
    sizes = [tuple(size) for size in sizes]
    if not sizes or any(len(size) != 2 or min(size) < 1 for size in sizes):
        raise ValueError("sizes must contain positive (height, width) pairs")
    image = gt.detach()
    if image.dtype in (torch.float16, torch.bfloat16):
        image = image.float()
    centered = image - image[:, :, :1, :1]
    gradients = []
    edges = []
    for size in sizes:
        resized = image if size == tuple(image.shape[-2:]) else F.interpolate(
            centered, size=size, mode="bilinear", align_corners=False, antialias=True
        )
        gradient = grad(resized)
        gradients.append(gradient)
        edges.append(edge_strength(gradient, scale=edge_scale))
    return {"gradient": gradients[0], "edge": edges[0],
            "gradient_pyramid": gradients, "edge_pyramid": edges}
