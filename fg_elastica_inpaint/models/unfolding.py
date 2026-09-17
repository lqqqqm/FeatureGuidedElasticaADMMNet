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


# 单轮 ADMM 使用的固定超参数；学习到的校正强度 alpha 由外层 network.py 管理。
@dataclass
class StageHyperParams:
    # r1/r2/r4 分别控制 |p|-m·p、p-grad(u)、n-m 三个约束的惩罚强度。
    r1: float = 1.0
    r2: float = 2.0
    r4: float = 1.0
    eta: float = 10.0  # 已知区域的数据保真权重。
    # 弹性正则中的 a 控制梯度长度项，b 控制以 div(n)^2 表示的曲率项。
    a: float = 0.1
    b: float = 0.2
    # u、n 子问题的内部梯度下降次数及步长；不等同于外层展开轮数 K。
    Tu: int = 3
    Tn: int = 3
    tau_u: float = 0.05
    tau_n: float = 0.125
    eps: float = 1e-6  # 向量归一化时的稳定项，避免零向量除零。

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
    # 固定 p 和 lambda2，用 Tu 步梯度下降近似求解 A u = b。
    # A=-r2*div(grad)+eta*M；M 使数据保真项只作用于已知区域。
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
    # p 是 [B,6,H,W] 的向量场；范数、散度和阈值按 RGB 各自计算，形状为 [B,3,H,W]。
    # c 为收缩阈值的分子，q 为由图像梯度、方向对齐项和乘子构成的收缩中心。
    c = a + b * div(n).square() + r1 + lambda1
    alignment = (r1 + lambda1) / r2  # [B, 3, H, W]
    # 复制三通道系数，让同一颜色的 x/y 分量使用相同的对齐权重。
    q = grad(u) + torch.cat([alignment, alignment], dim=1) * m - lambda2 / r2
    if rho_s < 0 or not math.isfinite(rho_s):
        raise ValueError("rho_s must be nonnegative")
    if rho_s > 0:
        if structure_gradient is None or M is None:
            raise ValueError("Positive rho_s requires a structure gradient and known mask")
        if structure_gradient.shape != q.shape or M.shape != u[:, :1].shape:
            raise ValueError("Structure gradient must match p and M must be [B,1,H,W]")
        # 只在孔洞内加入二次先验项：中心变为 q 与 G 的加权平均，阈值分母也随之增大。
        omega = rho_s * (1.0 - M)
        denominator = r2 + omega
        return vector_shrink((r2 * q + omega * structure_gradient) / denominator, c / denominator)
    # rho_s=0 时退化为原始向量收缩，按每个二维向量的模长缩放，保持其方向。
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
    # 将无约束更新 w 投影到逐像素、逐颜色的二维单位球，使 |m|<=1。
    # 球内向量保持原值；球外向量缩到单位长度，不是对所有向量强制单位化。
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
    # 固定 p、m 和 lambda4，求解带空间变化权重 mu=b*|p| 的 n 子问题。
    n = n_init
    mu = b * vector_norm(p)  # [B, 3, H, W], same shape as div(n)
    # ||-grad(mu*div)|| <= 8*max(mu) for this discrete grad/div pair.
    # A conservative step bound changes the numerical solver, not its equation.
    # 每个样本、每个颜色取空间最大权重，限制实际步长以提高迭代稳定性。
    bound = 8 * mu.amax(dim=(-2, -1), keepdim=True) + 0.5 * r4
    step = torch.minimum(torch.full_like(bound, tau_n), 0.9 / bound)
    step = torch.cat([step, step], dim=1)  # x/y 分量共享步长，并在 H、W 上广播。
    rhs = 0.5 * r4 * m - 0.5 * lambda4
    for _ in range(Tn):
        # mu 必须放在 grad 内，保留空间变化权重对算子的影响。
        An = -grad(mu * div(n)) + 0.5 * r4 * n
        n = n - step * (An - rhs)
    return n


# 一轮更新顺序：u -> p -> m -> n -> 乘子；u/p/n 可在解析更新后附加学习校正。
# tilde 表示子问题求解后的临时值，new 表示加入可选校正后传给下一步的值。
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

        # 融合多尺度语义；psi 将语义、变量、掩码和残差适配成 32 通道特征。
        self.fuse = MultiScaleFusion()
        self.psi_u = AdapterHead(128, 3, 3, 32)
        self.psi_p = AdapterHead(128, 6, 6, 32)
        self.psi_n = AdapterHead(128, 6, 6, 32)

        # phi 输出 gate*(base_correction+delta_res)，分别用于 3 通道图像和 6 通道向量场。
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

        # M=1 为已知区域；学习校正乘孔洞掩码，仅作用于 M=0 的区域。
        # 解析求解本身仍可更新已知区域，严格保留已知像素由外层 comp 合成完成。
        hole3 = (1.0 - M).repeat(1, u.shape[1], 1, 1)
        hole6 = (1.0 - M).repeat(1, p.shape[1], 1, 1)

        # 1. u 更新：有限步求解数据保真与梯度一致性子问题。
        # 关闭 use_unrolling 时跳过 ADMM 求解，但仍可执行启用的学习校正分支。
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

        # A*u_tilde-b 是 u 子问题的一阶最优性残差；其负值作为基础校正方向。
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

        # 2. p 更新：使用刚更新的 u，以及上一轮 m、n 和乘子进行向量收缩。
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
        # p 分支以 p=grad(u) 的一致性偏差作为基础校正。
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

        # 3. m 投影后立即用于 n 子问题；m 不设置独立的神经校正头。
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

        # 4. n 分支以 n=m 的一致性偏差为基础方向；启用校正时再对结果归一化。
        # 未启用校正时直接保留 n 子问题的输出，不额外归一化。
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

        # 5. 使用本轮校正后的变量更新乘子，分别累积三组约束残差。
        if self.use_unrolling:
            lambda1_new = lambda1 + self.hyper.r1 * (vector_norm(p_new) - dot_mp(m_new, p_new))
            lambda2_new = lambda2 + self.hyper.r2 * (p_new - grad(u_new))
            lambda4_new = lambda4 + self.hyper.r4 * (n_new - m_new)

        else:
            lambda1_new = lambda1
            lambda2_new = lambda2
            lambda4_new = lambda4

        # gate 为门控值，delta_res 为学习残差；实际增量还包含基础方向、alpha 和孔洞掩码。
        aux = {
            "correction_gate_u": u_aux["gate"],
            "correction_gate_p": p_aux["gate"],
            "correction_gate_n": n_aux["gate"],
            "correction_delta_u": u_aux["delta_res"],
            "correction_delta_p": p_aux["delta_res"],
            "correction_delta_n": n_aux["delta_res"],
        }
        # 以下指标仅用于观察迭代，不参与反向传播。
        with torch.no_grad():
            # 用同一 u_new 和旧 m/n/乘子重算 rho_s=0 的 p，衡量结构先验引入的变化。
            c = self.hyper.a + self.hyper.b * div(n).square() + self.hyper.r1 + lambda1
            alignment = (self.hyper.r1 + lambda1) / self.hyper.r2
            q = grad(u_new) + torch.cat([alignment, alignment], 1) * m - lambda2 / self.hyper.r2
            baseline_p = vector_shrink(q, c / self.hyper.r2)
            # 按整个 batch 的孔洞像素与通道求均值；无孔洞时通过分母下限避免除零。
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
            # 按需保留中间状态用于可视化或排查；detach 避免这些记录持有训练计算图。
            # q/c/p_baseline 对应未耦合的基线，threshold 则包含结构耦合后的分母。
            if return_state:
                aux["state"] = {"u_tilde": u_tilde.detach(), "u": u_new.detach(),
                    "p_tilde": p_tilde.detach(), "p": p_new.detach(), "m": m_new.detach(),
                    "n": n_new.detach(), "lambda1": lambda1_new.detach(),
                    "lambda2": lambda2_new.detach(), "lambda4": lambda4_new.detach(),
                    "q": q, "c": c, "threshold": c/(self.hyper.r2+rho_s*(1-M)),
                    "p_baseline": baseline_p, "div_n": div(n_new),
                    "correction_u": (u_new-u_tilde).detach()}
        return u_new, p_new, m_new, n_new, lambda1_new, lambda2_new, lambda4_new, aux
