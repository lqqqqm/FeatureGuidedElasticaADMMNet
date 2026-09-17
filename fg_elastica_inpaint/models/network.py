from __future__ import annotations

from typing import Any, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SemanticBackbone
from .operators import grad, normalize_vec
from .solvers import solve_u_pcg
from .structure_prior import LearnedStructurePrior
from .unfolding import StageHyperParams, UnfoldStage


# 整体流程：提取多尺度语义特征 -> 初始化 ADMM 状态 -> 展开 K 轮 -> 输出修复结果。
# 各轮复用同一个 UnfoldStage 的网络权重，但使用各自可学习的校正强度 alpha。
class FeatureGuidedElasticaADMMNet(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        K: int = 3,
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.0,
        use_mask_aware_transformer: bool = True,
        enable_u_correction: bool = True,
        enable_p_correction: bool = False,
        enable_n_correction: bool = False,
        use_unrolling: bool = True,
        use_softplus_alpha: bool = True,
        use_bounded_alpha: bool = True,
        alpha_u_init: float = 0.10,
        alpha_p_init: float = 0.00,
        alpha_n_init: float = 0.00,
        alpha_u_scale: float = 0.10,
        alpha_p_scale: float = 0.05,
        alpha_n_scale: float = 0.05,
        use_structure_prior: bool = False,
        structure_rho: float = 0.0,
        use_pcg_readout: bool = False,
        readout_iterations: int = 20,
        readout_tolerance: float = 1e-5,
        readout_backward_iterations: int = 80,
        stage_hyper: StageHyperParams | None = None,
    ):
        super().__init__()
        # 结构耦合是在 ADMM 的 p 子问题中引入梯度先验，因此需要同时启用这两个模块。
        if K < 1 or structure_rho < 0 or not math.isfinite(structure_rho):
            raise ValueError("K must be positive and structure_rho nonnegative")
        if structure_rho > 0 and (not use_structure_prior or not use_unrolling):
            raise ValueError("Structure coupling requires both the structure prior and ADMM unrolling")
        self.K = K
        self.structure_rho = float(structure_rho)
        self.use_pcg_readout = use_pcg_readout
        self.readout_iterations = readout_iterations
        self.readout_tolerance = readout_tolerance
        self.readout_backward_iterations = readout_backward_iterations
        self.use_softplus_alpha = use_softplus_alpha
        self.use_bounded_alpha = use_bounded_alpha
        self.alpha_scales = {
            "u": float(alpha_u_scale),
            "p": float(alpha_p_scale),
            "n": float(alpha_n_scale),
        }
        self.stage_hyper = stage_hyper or StageHyperParams()
        # RGB 图像与已知区域掩码拼成 4 通道输入；image_size 决定初始位置编码尺寸。
        self.backbone = SemanticBackbone(
            in_channels=4,
            base_hw=max(image_size // 8, 1),
            transformer_dim=512,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
            transformer_dropout=transformer_dropout,
            use_mask_aware_transformer=use_mask_aware_transformer,
        )
        self.stage = UnfoldStage(
            hyper=self.stage_hyper,
            enable_u_correction=enable_u_correction,
            enable_p_correction=enable_p_correction,
            enable_n_correction=enable_n_correction,
            use_unrolling=use_unrolling,
        )

        # 存储每轮校正强度的原始参数，使用时再经 _alpha 映射为实际系数。
        self.alpha_u = nn.Parameter(torch.full((K,), self._init_alpha_raw(alpha_u_init, alpha_u_scale)))
        self.alpha_p = nn.Parameter(torch.full((K,), self._init_alpha_raw(alpha_p_init, alpha_p_scale)))
        self.alpha_n = nn.Parameter(torch.full((K,), self._init_alpha_raw(alpha_n_init, alpha_n_scale)))
        self.eps = self.stage_hyper.eps
        # Initialize after the common model so R0/R1/R2 share backbone/stage
        # initialization when their seeds are equal.
        # 先初始化公共模块，再创建可选先验，保证同种子下消融实验的公共权重一致。
        self.structure_prior = LearnedStructurePrior() if use_structure_prior else None

    def _init_alpha_raw(self, value: float, scale: float) -> float:
        # 有界模式用 sigmoid 的逆变换初始化；截断比例避免 log(0) 或无穷值。
        # 无界模式直接存 value，若启用 softplus，实际初始强度为 softplus(value)。
        if not self.use_bounded_alpha:
            return float(value)
        scale = max(float(scale), 1e-6)
        ratio = min(max(float(value) / scale, 1e-4), 1.0 - 1e-4)
        return float(math.log(ratio / (1.0 - ratio)))

    def _alpha(self, p: torch.Tensor, idx: int, name: str) -> torch.Tensor:
        # 映射优先级：有界 sigmoid -> 正值 softplus -> 原始值。
        # scale 为正时，有界模式将强度限制在 0 与对应 scale 之间。
        if self.use_bounded_alpha:
            return self.alpha_scales[name] * torch.sigmoid(p[idx])
        if self.use_softplus_alpha:
            return F.softplus(p[idx])
        return p[idx]

    def forward(self, I_m: torch.Tensor, M: torch.Tensor, *, rho_scale: float = 1.0,
                return_stage_states: bool = False, prior_gradient: torch.Tensor | None = None,
                disable_correction: bool = False) -> Dict[str, Any]:
        # I_m: [B,3,H,W] 遮挡后的 RGB 图像；M: [B,1,H,W]，1 为已知、0 为待修复。
        # rho_scale 缩放结构耦合强度；prior_gradient 可覆盖网络预测的梯度先验。
        # return_stage_states 用于导出脱离计算图的中间状态，disable_correction 关闭学习校正。
        if I_m.ndim != 4 or I_m.shape[1] != 3 or M.shape != I_m[:, :1].shape:
            raise ValueError("Inputs must be RGB [B,3,H,W] and known mask [B,1,H,W]")
        if I_m.shape[-2] % 8 or I_m.shape[-1] % 8:
            raise ValueError("Image height and width must be multiples of 8")
        if rho_scale < 0 or not math.isfinite(rho_scale):
            raise ValueError("rho_scale must be finite and nonnegative")
        x = torch.cat([I_m, M], dim=1)
        # F1/F2/F3 的通道数分别为 64/128/256，分辨率分别为原图的 1、1/2、1/4。
        F1, F2, F3 = self.backbone(x)
        # 结构先验每张图只预测一次；外部梯度优先用于迭代，structure 仍保留网络输出。
        structure = self.structure_prior(F1, F2, F3, M) if self.structure_prior is not None else None
        G = prior_gradient if prior_gradient is not None else (structure["gradient"] if structure else None)
        # Backbone/structure CNNs may use autocast; all ADMM state is FP32.
        # 将状态、特征和先验转成 FP32，后续展开轮次也关闭 autocast 以稳定数值运算。
        I_m, M = I_m.float(), M.float()
        F1, F2, F3 = F1.float(), F2.float(), F3.float()
        G = G.float() if G is not None else None

        # u 为图像；p 为独立的梯度变量；m 为方向辅助变量；n 用于计算散度/曲率项。
        # p/m/n: [B,6,H,W]，前三通道为 RGB 的 x 分量，后三通道为 y 分量。
        u = I_m
        p = grad(u)
        m = normalize_vec(p, self.eps)
        n = normalize_vec(p, self.eps)

        B, _, H, W = I_m.shape
        # 三组乘子分别对应 |p|-m·p=0、p-grad(u)=0、n-m=0；首组每颜色一个标量。
        lambda1 = torch.zeros(B, 3, H, W, device=I_m.device, dtype=I_m.dtype)
        lambda2 = torch.zeros(B, 6, H, W, device=I_m.device, dtype=I_m.dtype)
        lambda4 = torch.zeros(B, 6, H, W, device=I_m.device, dtype=I_m.dtype)

        preds = []
        stage_aux = []
        # 按轮传递原始变量和乘子；每轮预测保留计算图，可用于多阶段监督。
        for k in range(self.K):
            with torch.autocast(device_type=I_m.device.type, enabled=False):
                u, p, m, n, lambda1, lambda2, lambda4, aux_k = self.stage(
                    u=u,
                    p=p,
                    m=m,
                    n=n,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    lambda4=lambda4,
                    I_m=I_m,
                    M=M,
                    F1=F1,
                    F2=F2,
                    F3=F3,
                    alpha_u=self._alpha(self.alpha_u, k, "u"),
                    alpha_p=self._alpha(self.alpha_p, k, "p"),
                    alpha_n=self._alpha(self.alpha_n, k, "n"),
                    structure_gradient=G,
                    rho_s=self.structure_rho * rho_scale,
                    return_state=return_stage_states,
                    disable_correction=disable_correction,
                )
            preds.append(u)
            stage_aux.append(aux_k)

        # 可选 PCG 读出在最后一轮的 p、lambda2 固定后，进一步求解 u 子问题。
        # stage_preds 记录读出前的各轮输出，最终 pred 则可能经过 PCG 更新。
        readout = {"u_before": u.detach()}
        if self.use_pcg_readout:
            u, info = solve_u_pcg(u, p, lambda2, I_m, M, self.stage_hyper.r2, self.stage_hyper.eta,
                                 self.readout_iterations, self.readout_tolerance, self.readout_backward_iterations)
            readout.update(info)
        # 二值掩码下，合成图 comp 严格保留已知像素；pred 是未经合成的求解结果。
        comp = M * I_m + (1.0 - M) * u
        structure_edge = structure["edge"] if structure else None
        structure_orient = normalize_vec(structure["gradient"].float(), self.eps) if structure else None
        last_aux = stage_aux[-1] if stage_aux else {}
        # 返回最终图像、各轮预测与诊断；aux 中的校正信息取自最后一轮。
        return {
            "pred": u,
            "comp": comp,
            "stage_preds": preds,
            "structure": structure,
            "readout": readout,
            "diagnostics": [item["diagnostics"] for item in stage_aux],
            "stage_states": [item["state"] for item in stage_aux] if return_stage_states else [],
            "aux": {
                "p": p,
                "m": m,
                "n": n,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "lambda4": lambda4,
                "structure_edge": structure_edge,
                "structure_orient": structure_orient,
                "correction_gate_u": last_aux.get("correction_gate_u"),
                "correction_gate_p": last_aux.get("correction_gate_p"),
                "correction_gate_n": last_aux.get("correction_gate_n"),
                "correction_delta_u": last_aux.get("correction_delta_u"),
                "correction_delta_p": last_aux.get("correction_delta_p"),
                "correction_delta_n": last_aux.get("correction_delta_n"),
            },
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "FeatureGuidedElasticaADMMNet":
        # 将 data/model/stage_hyper 三组配置映射为构造参数，缺省项采用这里的默认值。
        model_cfg = cfg["model"]
        stage_cfg = cfg.get("stage_hyper", {})
        hyper = StageHyperParams(
            r1=stage_cfg.get("r1", 1.0),
            r2=stage_cfg.get("r2", 2.0),
            r4=stage_cfg.get("r4", 1.0),
            eta=stage_cfg.get("eta", 10.0),
            a=stage_cfg.get("a", 0.1),
            b=stage_cfg.get("b", 0.2),
            Tu=stage_cfg.get("Tu", 3),
            Tn=stage_cfg.get("Tn", 3),
            tau_u=stage_cfg.get("tau_u", 0.05),
            tau_n=stage_cfg.get("tau_n", 0.125),
            eps=stage_cfg.get("eps", 1e-6),
        )
        return cls(
            image_size=cfg["data"].get("image_size", 256),
            K=model_cfg.get("K", 3),
            transformer_depth=model_cfg.get("transformer_depth", 2),
            transformer_heads=model_cfg.get("transformer_heads", 8),
            transformer_mlp_ratio=model_cfg.get("transformer_mlp_ratio", 4.0),
            transformer_dropout=model_cfg.get("transformer_dropout", 0.0),
            use_mask_aware_transformer=model_cfg.get("use_mask_aware_transformer", True),
            enable_u_correction=model_cfg.get("enable_u_correction", True),
            enable_p_correction=model_cfg.get("enable_p_correction", False),
            enable_n_correction=model_cfg.get("enable_n_correction", False),
            use_unrolling=model_cfg.get("use_unrolling", True),
            use_softplus_alpha=model_cfg.get("use_softplus_alpha", True),
            use_bounded_alpha=model_cfg.get("use_bounded_alpha", True),
            alpha_u_init=model_cfg.get("alpha_u_init", 0.10),
            alpha_p_init=model_cfg.get("alpha_p_init", 0.00),
            alpha_n_init=model_cfg.get("alpha_n_init", 0.00),
            alpha_u_scale=model_cfg.get("alpha_u_scale", 0.10),
            alpha_p_scale=model_cfg.get("alpha_p_scale", 0.05),
            alpha_n_scale=model_cfg.get("alpha_n_scale", 0.05),
            use_structure_prior=model_cfg.get("use_structure_prior", False),
            structure_rho=model_cfg.get("structure_rho", 0.0),
            use_pcg_readout=model_cfg.get("use_pcg_readout", False),
            readout_iterations=model_cfg.get("readout_iterations", 20),
            readout_tolerance=model_cfg.get("readout_tolerance", 1e-5),
            readout_backward_iterations=model_cfg.get("readout_backward_iterations", 80),
            stage_hyper=hyper,
        )
