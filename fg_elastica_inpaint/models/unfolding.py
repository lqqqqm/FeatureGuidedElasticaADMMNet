from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch import Tensor

from .adapters import AdapterHead, MultiScaleFusion
from .corrections import CorrectionHead
from .operators import (
    div,
    dot_mp,
    grad,
    laplace,
    normalize_vec,
    proj_unit_ball,
    vector_norm,
    vector_shrink,
)


@dataclass
class StageHyperParams:
    r1: float = 1.0
    r2: float = 2.0
    r4: float = 1.0
    eta: float = 10.0
    a: float = 0.1
    b: float = 0.2
    Tu: int = 3
    Tn: int = 3
    tau_u: float = 0.05
    tau_n: float = 0.125
    eps: float = 1e-6

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.r1, self.r2, self.r4, self.eta, self.a, self.b))
                or min(self.r1, self.r2, self.r4, self.eta) <= 0 or min(self.a, self.b) < 0):
            raise ValueError("ADMM penalties/eta must be positive and a,b nonnegative")



def solve_u_gd(
    u_init: Tensor,
    p: Tensor,
    lambda2: Tensor,
    I_m: Tensor,
    M: Tensor,
    r2: float,
    eta: float,
    Tu: int = 3,
    tau_u: float = 0.05,
) -> Tensor:
    u = u_init
    for _ in range(Tu):
        Au = -r2 * laplace(u) + eta * M * u
        bu = eta * M * I_m - div(r2 * p + lambda2)
        u = u - tau_u * (Au - bu)
    return u



def solve_p_shrink(
    u: Tensor,
    m: Tensor,
    n: Tensor,
    lambda1: Tensor,
    lambda2: Tensor,
    a: float,
    b: float,
    r1: float,
    r2: float,
    structure_gradient: Tensor | None = None,
    M: Tensor | None = None,
    rho_s: float = 0.0,
) -> Tensor:
    """Derivation 4.2 plus rho_s*(1-M)*|p-G|^2/2, per RGB vector."""
    c = a + b * div(n).square() + r1 + lambda1
    alignment = (r1 + lambda1) / r2  # [B, 3, H, W]
    q = grad(u) + torch.cat([alignment, alignment], dim=1) * m - lambda2 / r2
    if rho_s < 0 or not math.isfinite(rho_s):
        raise ValueError("rho_s must be nonnegative")
    if rho_s > 0:
        if structure_gradient is None or M is None:
            raise ValueError("Positive rho_s requires a structure gradient and known mask")
        if structure_gradient.shape != q.shape or M.shape != u[:, :1].shape:
            raise ValueError("Structure gradient must match p and M must be [B,1,H,W]")
        omega = rho_s * (1.0 - M)
        denominator = r2 + omega
        return vector_shrink((r2 * q + omega * structure_gradient) / denominator, c / denominator)
    return vector_shrink(q, c / r2)



def solve_m_proj(
    p: Tensor,
    n: Tensor,
    lambda1: Tensor,
    lambda4: Tensor,
    r1: float,
    r4: float,
) -> Tensor:
    """Derivation 4.3: project n + (r1+lambda1)p/r4 + lambda4/r4."""
    alignment = (r1 + lambda1) / r4
    w = n + torch.cat([alignment, alignment], dim=1) * p + lambda4 / r4
    return proj_unit_ball(w)



def solve_n_gd(
    n_init: Tensor,
    p: Tensor,
    m: Tensor,
    lambda4: Tensor,
    r4: float,
    b: float,
    Tn: int = 3,
    tau_n: float = 0.125,
) -> Tensor:
    """GD for derivation 4.4, with the full weighted grad-div operator."""
    n = n_init
    mu = b * vector_norm(p)  # [B, 3, H, W], same shape as div(n)
    # ||-grad(mu*div)|| <= 8*max(mu) for this discrete grad/div pair.
    # A conservative step bound changes the numerical solver, not its equation.
    bound = 8 * mu.amax(dim=(-2, -1), keepdim=True) + 0.5 * r4
    step = torch.minimum(torch.full_like(bound, tau_n), 0.9 / bound)
    step = torch.cat([step, step], dim=1)
    rhs = 0.5 * r4 * m - 0.5 * lambda4
    for _ in range(Tn):
        An = -grad(mu * div(n)) + 0.5 * r4 * n
        n = n - step * (An - rhs)
    return n


class UnfoldStage(nn.Module):
    def __init__(
        self,
        hyper: StageHyperParams,
        enable_u_correction: bool = True,
        enable_p_correction: bool = False,
        enable_n_correction: bool = False,
        use_unrolling: bool = True,
    ):
        super().__init__()
        self.hyper = hyper
        self.enable_u_correction = enable_u_correction
        self.enable_p_correction = enable_p_correction
        self.enable_n_correction = enable_n_correction
        self.use_unrolling = use_unrolling

        self.fuse = MultiScaleFusion()
        self.psi_u = AdapterHead(128, 3, 3, 32)
        self.psi_p = AdapterHead(128, 6, 6, 32)
        self.psi_n = AdapterHead(128, 6, 6, 32)

        self.phi_u = CorrectionHead(3, 32, 3)
        self.phi_p = CorrectionHead(6, 32, 6)
        self.phi_n = CorrectionHead(6, 32, 6)

    def forward(
        self,
        u: Tensor,
        p: Tensor,
        m: Tensor,
        n: Tensor,
        lambda1: Tensor,
        lambda2: Tensor,
        lambda4: Tensor,
        I_m: Tensor,
        M: Tensor,
        F1: Tensor,
        F2: Tensor,
        F3: Tensor,
        alpha_u: Tensor,
        alpha_p: Tensor,
        alpha_n: Tensor,
        structure_gradient: Tensor | None = None,
        rho_s: float = 0.0,
        return_state: bool = False,
        disable_correction: bool = False,
    ):
        F_ms = self.fuse(F1, F2, F3)

        hole3 = (1.0 - M).repeat(1, u.shape[1], 1, 1)
        hole6 = (1.0 - M).repeat(1, p.shape[1], 1, 1)

        if self.use_unrolling:
            u_tilde = solve_u_gd(
                u_init=u,
                p=p,
                lambda2=lambda2,
                I_m=I_m,
                M=M,
                r2=self.hyper.r2,
                eta=self.hyper.eta,
                Tu=self.hyper.Tu,
                tau_u=self.hyper.tau_u,
            )
        else:
            u_tilde = u

        u_stationarity = -self.hyper.r2 * laplace(u_tilde) + self.hyper.eta * M * u_tilde
        u_stationarity = u_stationarity - (
            self.hyper.eta * M * I_m - div(self.hyper.r2 * p + lambda2)
        )
        base_u = -u_stationarity
        F_u = self.psi_u(
            semantic=F_ms,
            variable=u_tilde,
            mask=M,
            residual=base_u,
        )
        u_aux = {"delta_res": None, "gate": None}
        if self.enable_u_correction and not disable_correction:
            du, u_aux = self.phi_u(
                variable=u_tilde,
                adapted_feature=F_u,
                mask=M,
                base_correction=base_u,
                return_aux=True,
            )
            u_new = u_tilde + alpha_u * hole3 * du
        else:
            u_new = u_tilde

        if self.use_unrolling:
            p_tilde = solve_p_shrink(
                u=u_new,
                m=m,
                n=n,
                lambda1=lambda1,
                lambda2=lambda2,
                a=self.hyper.a,
                b=self.hyper.b,
                r1=self.hyper.r1,
                r2=self.hyper.r2,
                structure_gradient=structure_gradient,
                M=M,
                rho_s=rho_s,
            )
        else:
            p_tilde = p
        base_p = grad(u_new) - p_tilde
        F_p = self.psi_p(
            semantic=F_ms,
            variable=p_tilde,
            mask=M,
            residual=base_p,
        )
        p_aux = {"delta_res": None, "gate": None}
        if self.enable_p_correction and not disable_correction:
            dp, p_aux = self.phi_p(
                variable=p_tilde,
                adapted_feature=F_p,
                mask=M,
                base_correction=base_p,
                return_aux=True,
            )
            p_new = p_tilde + alpha_p * hole6 * dp
        else:
            p_new = p_tilde

        if self.use_unrolling:
            m_new = solve_m_proj(
                p=p_new,
                n=n,
                lambda1=lambda1,
                lambda4=lambda4,
                r1=self.hyper.r1,
                r4=self.hyper.r4,
            )
            n_tilde = solve_n_gd(
                n_init=n,
                p=p_new,
                m=m_new,
                lambda4=lambda4,
                r4=self.hyper.r4,
                b=self.hyper.b,
                Tn=self.hyper.Tn,
                tau_n=self.hyper.tau_n,
            )
        else:
            m_new = normalize_vec(p_new, self.hyper.eps)
            n_tilde = n

        base_n = m_new - n_tilde
        F_n = self.psi_n(
            semantic=F_ms,
            variable=n_tilde,
            mask=M,
            residual=base_n,
        )
        n_aux = {"delta_res": None, "gate": None}
        if self.enable_n_correction and not disable_correction:
            dn, n_aux = self.phi_n(
                variable=n_tilde,
                adapted_feature=F_n,
                mask=M,
                base_correction=base_n,
                return_aux=True,
            )
            n_new = normalize_vec(n_tilde + alpha_n * hole6 * dn, self.hyper.eps)
        else:
            n_new = n_tilde

        if self.use_unrolling:
            lambda1_new = lambda1 + self.hyper.r1 * (vector_norm(p_new) - dot_mp(m_new, p_new))
            lambda2_new = lambda2 + self.hyper.r2 * (p_new - grad(u_new))
            lambda4_new = lambda4 + self.hyper.r4 * (n_new - m_new)

        else:
            lambda1_new = lambda1
            lambda2_new = lambda2
            lambda4_new = lambda4

        aux = {
            "correction_gate_u": u_aux["gate"],
            "correction_gate_p": p_aux["gate"],
            "correction_gate_n": n_aux["gate"],
            "correction_delta_u": u_aux["delta_res"],
            "correction_delta_p": p_aux["delta_res"],
            "correction_delta_n": n_aux["delta_res"],
        }
        with torch.no_grad():
            c = self.hyper.a + self.hyper.b * div(n).square() + self.hyper.r1 + lambda1
            alignment = (self.hyper.r1 + lambda1) / self.hyper.r2
            q = grad(u_new) + torch.cat([alignment, alignment], 1) * m - lambda2 / self.hyper.r2
            baseline_p = vector_shrink(q, c / self.hyper.r2)
            hole_mean = lambda x: (x * (1-M)).sum() / ((1-M).sum() * x.shape[1]).clamp_min(1)
            aux["diagnostics"] = {
                "rho_s": u.new_tensor(rho_s),
                "prior_fraction": u.new_tensor(rho_s / (self.hyper.r2 + rho_s)),
                "p_nonzero_hole": hole_mean((vector_norm(p_tilde) > 1e-6).float()),
                "p_injection_l1": hole_mean((p_tilde-baseline_p).abs()) if self.use_unrolling else u.new_zeros(()),
                "p_constraint": hole_mean((p_new-grad(u_new)).abs()),
                "m_constraint": hole_mean((vector_norm(p_new)-dot_mp(m_new,p_new)).abs()),
                "n_constraint": hole_mean((n_new-m_new).abs()),
                "curvature_mean": hole_mean(div(n_new).abs()),
                "correction_u_l1": hole_mean((u_new-u_tilde).abs()),
            }
            if structure_gradient is not None:
                aux["diagnostics"]["p_prior_l1"] = hole_mean((p_new-structure_gradient).abs())
            if return_state:
                aux["state"] = {"u_tilde": u_tilde.detach(), "u": u_new.detach(),
                    "p_tilde": p_tilde.detach(), "p": p_new.detach(), "m": m_new.detach(),
                    "n": n_new.detach(), "lambda1": lambda1_new.detach(),
                    "lambda2": lambda2_new.detach(), "lambda4": lambda4_new.detach(),
                    "q": q, "c": c, "threshold": c/(self.hyper.r2+rho_s*(1-M)),
                    "p_baseline": baseline_p, "div_n": div(n_new),
                    "correction_u": (u_new-u_tilde).detach()}
        return u_new, p_new, m_new, n_new, lambda1_new, lambda2_new, lambda4_new, aux
