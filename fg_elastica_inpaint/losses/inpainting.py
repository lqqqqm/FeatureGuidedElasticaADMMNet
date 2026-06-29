from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.operators import grad
from .perceptual import VGGPerceptualLoss


class InpaintingLoss(nn.Module):
    def __init__(
        self,
        K: int,
        lambda_rec: float = 1.0,
        lambda_perc: float = 0.0,
        lambda_edge: float = 1.0,
        lambda_stage: float = 0.5,
        hole_weight: float = 6.0,
        edge_hole_only: bool = True,
        use_perceptual: bool = False,
        vgg_pretrained: bool = False,
        stage_weights: Optional[List[float]] = None,
        lambda_p_cons: float = 0.0,
        lambda_n_m: float = 0.0,
        lambda_struct: float = 0.0,
    ):
        super().__init__()
        self.K = K
        self.lambda_rec = lambda_rec
        self.lambda_perc = lambda_perc
        self.lambda_edge = lambda_edge
        self.lambda_stage = lambda_stage
        self.hole_weight = hole_weight
        self.edge_hole_only = edge_hole_only
        self.lambda_p_cons = lambda_p_cons
        self.lambda_n_m = lambda_n_m
        self.lambda_struct = lambda_struct
        self.perc = VGGPerceptualLoss(pretrained=vgg_pretrained) if use_perceptual else None
        if stage_weights is None:
            stage_weights = [(i + 1) / sum(range(1, K + 1)) for i in range(K)]
        if len(stage_weights) != K:
            raise ValueError(f"stage_weights length must equal K={K}, got {len(stage_weights)}")
        self.stage_weights = stage_weights

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (x * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, outputs: Dict, gt: torch.Tensor, M: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = outputs["pred"]
        stage_preds = outputs["stage_preds"]
        M3 = M.repeat(1, 3, 1, 1)
        hole3 = 1.0 - M3
        hole6 = 1.0 - M.repeat(1, 6, 1, 1)

        l_valid = self.masked_mean((pred - gt).abs(), M3)
        l_hole = self.masked_mean((pred - gt).abs(), hole3)
        l_rec = l_valid + self.hole_weight * l_hole

        edge_diff = (grad(pred) - grad(gt)).abs()
        if self.edge_hole_only:
            l_edge = self.masked_mean(edge_diff, hole6)
        else:
            l_edge = edge_diff.mean()

        l_perc = pred.new_tensor(0.0)
        if self.perc is not None:
            comp = M3 * gt + hole3 * pred
            l_perc = self.perc(comp, gt)

        l_stage = pred.new_tensor(0.0)
        for w, stage_pred in zip(self.stage_weights, stage_preds):
            l_stage = l_stage + w * F.l1_loss(stage_pred, gt)

        aux = outputs.get("aux", {})
        l_p_cons = pred.new_tensor(0.0)
        p = aux.get("p")
        if p is not None and self.lambda_p_cons > 0.0:
            l_p_cons = self.masked_mean((p - grad(pred)).abs(), hole6)

        l_n_m = pred.new_tensor(0.0)
        n = aux.get("n")
        m = aux.get("m")
        if n is not None and m is not None and self.lambda_n_m > 0.0:
            l_n_m = self.masked_mean((n - m).abs(), hole6)

        l_struct = pred.new_tensor(0.0)
        edge_pred = aux.get("structure_edge")
        if edge_pred is not None and self.lambda_struct > 0.0:
            edge_gt = self._edge_target(gt)
            l_struct = self.masked_mean((edge_pred - edge_gt).abs(), 1.0 - M)

        total = (
            self.lambda_rec * l_rec
            + self.lambda_perc * l_perc
            + self.lambda_edge * l_edge
            + self.lambda_stage * l_stage
            + self.lambda_p_cons * l_p_cons
            + self.lambda_n_m * l_n_m
            + self.lambda_struct * l_struct
        )
        return {
            "total": total,
            "rec": l_rec.detach(),
            "edge": l_edge.detach(),
            "perc": l_perc.detach(),
            "stage": l_stage.detach(),
            "p_cons": l_p_cons.detach(),
            "n_m": l_n_m.detach(),
            "struct": l_struct.detach(),
        }

    @staticmethod
    def _edge_target(gt: torch.Tensor) -> torch.Tensor:
        g = grad(gt)
        c = gt.shape[1]
        gx, gy = g[:, :c], g[:, c:]
        mag = (gx.square() + gy.square() + 1e-6).sqrt().mean(dim=1, keepdim=True)
        return mag / mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
