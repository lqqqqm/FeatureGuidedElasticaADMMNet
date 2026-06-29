from __future__ import annotations

from typing import Any, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SemanticBackbone
from .operators import grad, normalize_vec
from .unfolding import StageHyperParams, UnfoldStage


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
        use_adaptive_beta: bool = False,
        use_adaptive_mu: bool = False,
        use_structure_head: bool = False,
        structure_gamma: float = 0.0,
        stage_hyper: StageHyperParams | None = None,
    ):
        super().__init__()
        self.K = K
        self.use_softplus_alpha = use_softplus_alpha
        self.use_bounded_alpha = use_bounded_alpha
        self.alpha_scales = {
            "u": float(alpha_u_scale),
            "p": float(alpha_p_scale),
            "n": float(alpha_n_scale),
        }
        self.stage_hyper = stage_hyper or StageHyperParams()
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
            use_adaptive_beta=use_adaptive_beta,
            use_adaptive_mu=use_adaptive_mu,
            use_structure_head=use_structure_head,
            structure_gamma=structure_gamma,
        )

        self.alpha_u = nn.Parameter(torch.full((K,), self._init_alpha_raw(alpha_u_init, alpha_u_scale)))
        self.alpha_p = nn.Parameter(torch.full((K,), self._init_alpha_raw(alpha_p_init, alpha_p_scale)))
        self.alpha_n = nn.Parameter(torch.full((K,), self._init_alpha_raw(alpha_n_init, alpha_n_scale)))
        self.eps = self.stage_hyper.eps

    def _init_alpha_raw(self, value: float, scale: float) -> float:
        if not self.use_bounded_alpha:
            return float(value)
        scale = max(float(scale), 1e-6)
        ratio = min(max(float(value) / scale, 1e-4), 1.0 - 1e-4)
        return float(math.log(ratio / (1.0 - ratio)))

    def _alpha(self, p: torch.Tensor, idx: int, name: str) -> torch.Tensor:
        if self.use_bounded_alpha:
            return self.alpha_scales[name] * torch.sigmoid(p[idx])
        if self.use_softplus_alpha:
            return F.softplus(p[idx])
        return p[idx]

    def forward(self, I_m: torch.Tensor, M: torch.Tensor) -> Dict[str, Any]:
        x = torch.cat([I_m, M], dim=1)
        F1, F2, F3 = self.backbone(x)

        u = I_m
        p = grad(u)
        m = normalize_vec(p, self.eps)
        n = normalize_vec(p, self.eps)

        B, _, H, W = I_m.shape
        lambda1 = torch.zeros(B, 3, H, W, device=I_m.device, dtype=I_m.dtype)
        lambda2 = torch.zeros(B, 6, H, W, device=I_m.device, dtype=I_m.dtype)
        lambda4 = torch.zeros(B, 6, H, W, device=I_m.device, dtype=I_m.dtype)

        preds = []
        stage_aux = []
        for k in range(self.K):
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
            )
            preds.append(u)
            stage_aux.append(aux_k)

        comp = M * I_m + (1.0 - M) * u
        structure_edge = next((item["structure_edge"] for item in reversed(stage_aux) if item["structure_edge"] is not None), None)
        structure_orient = next((item["structure_orient"] for item in reversed(stage_aux) if item["structure_orient"] is not None), None)
        last_aux = stage_aux[-1] if stage_aux else {}
        return {
            "pred": u,
            "comp": comp,
            "stage_preds": preds,
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
        model_cfg = cfg["model"]
        stage_cfg = cfg.get("stage_hyper", {})
        hyper = StageHyperParams(
            r1=stage_cfg.get("r1", 1.0),
            r2=stage_cfg.get("r2", 2.0),
            r4=stage_cfg.get("r4", 1.0),
            eta=stage_cfg.get("eta", 10.0),
            mu0=stage_cfg.get("mu0", 0.2),
            beta_p=stage_cfg.get("beta_p", 0.1),
            gamma_p=stage_cfg.get("gamma_p", 0.5),
            gamma_n=stage_cfg.get("gamma_n", 0.5),
            Tu=stage_cfg.get("Tu", 3),
            Tn=stage_cfg.get("Tn", 3),
            tau_u=stage_cfg.get("tau_u", 0.05),
            tau_n=stage_cfg.get("tau_n", 0.125),
            eps=stage_cfg.get("eps", 1e-6),
            lambda_max=stage_cfg.get("lambda_max", 10.0),
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
            use_adaptive_beta=model_cfg.get("use_adaptive_beta", False),
            use_adaptive_mu=model_cfg.get("use_adaptive_mu", False),
            use_structure_head=model_cfg.get("use_structure_head", False),
            structure_gamma=model_cfg.get("structure_gamma", 0.0),
            stage_hyper=hyper,
        )
