from __future__ import annotations

import torch
from torch import Tensor

from .operators import div, grad


def poisson_operator(u: Tensor, M: Tensor, r2: float, eta: float) -> Tensor:
    return -r2 * div(grad(u)) + eta * M * u


def _pcg(rhs: Tensor, M: Tensor, initial: Tensor, r2: float, eta: float,
         iterations: int, tolerance: float) -> Tensor:
    # Diagonal of -div(grad): boundary pixels have fewer neighbours.
    degree = torch.zeros_like(M)
    degree[..., :-1] += 1
    degree[..., 1:] += 1
    degree[..., :-1, :] += 1
    degree[..., 1:, :] += 1
    diagonal = r2 * degree + eta * M
    tiny = torch.finfo(rhs.dtype).tiny
    dot = lambda x, y: (x * y).sum(dim=(-2, -1), keepdim=True)
    u = initial.clone()
    residual = rhs - poisson_operator(u, M, r2, eta)
    z = residual / diagonal.clamp_min(tiny)
    direction = z.clone()
    rz = dot(residual, z)
    # Relative stopping is essential for the small RHS of mean-loss VJPs.
    # An absolute floor of one would make gradients depend on loss scaling.
    threshold = tolerance**2 * dot(rhs, rhs)
    for _ in range(iterations):
        active = dot(residual, residual) > threshold
        if not bool(active.any()):
            break
        Ad = poisson_operator(direction, M, r2, eta)
        denominator = dot(direction, Ad)
        alpha = torch.where(active, rz / denominator.clamp_min(tiny), 0.)
        u = u + alpha * direction
        residual = residual - alpha * Ad
        z = residual / diagonal.clamp_min(tiny)
        rz_new = dot(residual, z)
        beta = torch.where(active, rz_new / rz.clamp_min(tiny), 0.)
        direction = z + beta * direction
        rz = rz_new
    return u


class _PoissonSolve(torch.autograd.Function):
    """Implicit derivative of A u=b; backward solves A^T v=dL/du.

    A is symmetric. Gradients approximate the equation's derivative when the
    finite PCG budgets leave residuals; they are not derivatives of CG control
    flow. This also gives the correct derivative for an exactly zero RHS.
    """

    @staticmethod
    def forward(ctx, rhs, M, initial, r2, eta, iterations, tolerance, backward_iterations, info):
        u = _pcg(rhs, M, initial, r2, eta, iterations, tolerance)
        ctx.save_for_backward(M, u)
        ctx.settings = (r2, eta, backward_iterations, tolerance)
        ctx.info = info
        return u

    @staticmethod
    def backward(ctx, gradient):
        M, u = ctx.saved_tensors
        r2, eta, iterations, tolerance = ctx.settings
        adjoint = _pcg(gradient, M, torch.zeros_like(gradient), r2, eta, iterations, tolerance)
        with torch.no_grad():
            residual = gradient - poisson_operator(adjoint, M, r2, eta)
            relative = torch.linalg.vector_norm(residual, dim=(-2, -1)) / torch.linalg.vector_norm(
                gradient, dim=(-2, -1)).clamp_min(torch.finfo(gradient.dtype).tiny)
            ctx.info["backward_relative_residual"].copy_(relative)
            ctx.info["backward_solved"].fill_(True)
        grad_M = (-eta * adjoint * u).sum(1, keepdim=True) if ctx.needs_input_grad[1] else None
        return adjoint, grad_M, torch.zeros_like(u), None, None, None, None, None, None


def solve_u_pcg(u_init: Tensor, p: Tensor, lambda2: Tensor, I_m: Tensor, M: Tensor,
                r2: float, eta: float, iterations: int = 20, tolerance: float = 1e-5,
                backward_iterations: int = 80) -> tuple[Tensor, dict]:
    """Solve the original u equation with a boundary-correct Jacobi PCG readout."""
    if r2 <= 0 or eta <= 0 or iterations < 1 or backward_iterations < 1 or tolerance <= 0:
        raise ValueError("PCG requires positive r2, eta, iteration budgets and tolerance")
    if bool((M.sum((-2, -1)) <= 0).any()):
        raise ValueError("The Neumann u equation needs at least one known pixel per image")
    dtype = torch.float64 if u_init.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=u_init.device.type, enabled=False):
        u_init, p, lambda2, I_m, M = [x.to(dtype) for x in (u_init, p, lambda2, I_m, M)]
        rhs = eta * M * I_m - div(r2 * p + lambda2)
        info = {"backward_relative_residual": rhs.new_zeros(rhs.shape[:2]),
                "backward_solved": torch.tensor(False, device=rhs.device)}
        result = _PoissonSolve.apply(rhs, M, u_init, r2, eta, iterations, tolerance, backward_iterations, info)
        with torch.no_grad():
            residual = rhs - poisson_operator(result, M, r2, eta)
            denominator = torch.linalg.vector_norm(rhs, dim=(-2, -1)).clamp_min(1e-12)
            absolute = torch.linalg.vector_norm(residual, dim=(-2, -1))
            relative = absolute / denominator
        info.update(relative_residual=relative, absolute_residual=absolute)
        return result, info
