from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def grad(u: Tensor) -> Tensor:
    """Forward difference with zero normal derivative at the image boundary.

    Args:
        u: [B, C, H, W]
    Returns:
        [B, 2C, H, W] where first C channels are dx and last C channels are dy.
    """
    ux_pad = F.pad(u, (0, 1, 0, 0), mode="replicate")
    uy_pad = F.pad(u, (0, 0, 0, 1), mode="replicate")
    ux = ux_pad[..., 1:] - ux_pad[..., :-1]
    uy = uy_pad[:, :, 1:, :] - uy_pad[:, :, :-1, :]
    return torch.cat([ux, uy], dim=1)


def div(v: Tensor) -> Tensor:
    """Negative adjoint of grad(): <grad(u), v> = -<u, div(v)>.

    Args:
        v: [B, 2C, H, W]
    Returns:
        [B, C, H, W]
    """
    c = v.shape[1] // 2
    vx, vy = v[:, :c], v[:, c:]
    # The final forward differences are zero; their dual components do not
    # contribute. Zero padding supplies the matching boundary fluxes.
    vx_pad = F.pad(vx[..., :-1], (1, 1, 0, 0))
    vy_pad = F.pad(vy[:, :, :-1, :], (0, 0, 1, 1))
    dx = vx_pad[..., 1:] - vx_pad[..., :-1]
    dy = vy_pad[:, :, 1:, :] - vy_pad[:, :, :-1, :]
    return dx + dy


def laplace(u: Tensor) -> Tensor:
    return div(grad(u))



def vector_norm(z: Tensor, eps: float = 0.0) -> Tensor:
    c = z.shape[1] // 2
    zx, zy = z[:, :c], z[:, c:]
    if eps:
        return (zx.square() + zy.square() + eps).sqrt()
    # Exact |z|, with PyTorch's finite zero subgradient at z=0.
    return torch.linalg.vector_norm(torch.stack([zx, zy], dim=0), dim=0)


def normalize_vec(z: Tensor, eps: float = 1e-6) -> Tensor:
    n = vector_norm(z, eps)
    n = torch.cat([n, n], dim=1)
    return z / n


def dot_mp(m: Tensor, p: Tensor) -> Tensor:
    c = m.shape[1] // 2
    mx, my = m[:, :c], m[:, c:]
    px, py = p[:, :c], p[:, c:]
    return mx * px + my * py


def proj_unit_ball(w: Tensor) -> Tensor:
    n = vector_norm(w)
    denom = torch.maximum(torch.ones_like(n), n)
    denom = torch.cat([denom, denom], dim=1)
    return w / denom


def _broadcast_tau(tau: float | Tensor, z: Tensor) -> Tensor:
    if not torch.is_tensor(tau):
        tau_t = torch.tensor(float(tau), dtype=z.dtype, device=z.device)
        return tau_t
    if tau.ndim == 0:
        return tau.to(dtype=z.dtype, device=z.device)
    if tau.ndim != 4:
        raise ValueError(f"tau must be scalar or [B,C,H,W], got shape={tuple(tau.shape)}")
    if tau.shape[1] * 2 == z.shape[1]:
        tau = torch.cat([tau, tau], dim=1)
    return tau.to(dtype=z.dtype, device=z.device)


def vector_shrink(q: Tensor, tau: float | Tensor) -> Tensor:
    n = vector_norm(q)
    # Evaluate the exact shrinkage for |q|>0 and return zero at q=0.
    denom = torch.where(n > 0, n, torch.ones_like(n))
    if torch.is_tensor(tau) and tau.ndim == 4 and tau.shape[1] == n.shape[1]:
        tau_n = tau.to(dtype=q.dtype, device=q.device)
    else:
        tau_n = _broadcast_tau(tau, q)
        if not torch.is_tensor(tau_n) or tau_n.ndim == 0:
            coef = torch.clamp(1.0 - tau_n / denom, min=0.0)
            coef = torch.where((n == 0) & (tau_n > 0), 0.0, coef)
            coef = torch.cat([coef, coef], dim=1)
            return coef * q
        tau_n = tau_n[:, : n.shape[1]]
    coef = torch.clamp(1.0 - tau_n / denom, min=0.0)
    coef = torch.where((n == 0) & (tau_n > 0), 0.0, coef)
    coef = torch.cat([coef, coef], dim=1)
    return coef * q
