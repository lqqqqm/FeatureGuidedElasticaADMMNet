from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.operators import grad
from ..utils.structure import edge_strength
from .perceptual import VGGPerceptualLoss
from .structure import structure_losses


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
        lambda_structure_grad: float = 0.0,
        lambda_structure_edge: float = 0.0,
        lambda_structure_orientation: float = 0.0,
        lambda_structure_consistency: float = 0.0,
        edge_scale: float = 0.1,
        orientation_threshold: float = 0.05,
        structure_scale_weights: Optional[List[float]] = None,
        structure_known_weight: float = 0.1,
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
        self.lambda_structure_grad = lambda_structure_grad
        self.lambda_structure_edge = lambda_structure_edge
        self.lambda_structure_orientation = lambda_structure_orientation
        self.lambda_structure_consistency = lambda_structure_consistency
        self.structure_loss_weights = {
            "structure_grad": lambda_structure_grad,
            "structure_edge": lambda_structure_edge,
            "structure_orientation": lambda_structure_orientation,
            "structure_consistency": lambda_structure_consistency,
        }
        if any(not math.isfinite(w) or w < 0 for w in self.structure_loss_weights.values()):
            raise ValueError("structure loss weights must be finite and nonnegative")
        if not math.isfinite(edge_scale) or edge_scale <= 0:
            raise ValueError("edge_scale must be finite and positive")
        if not math.isfinite(orientation_threshold) or orientation_threshold < 0:
            raise ValueError("orientation_threshold must be finite and nonnegative")
        if not math.isfinite(structure_known_weight) or not 0 <= structure_known_weight <= 1:
            raise ValueError("structure_known_weight must be between zero and one")
        if structure_scale_weights is not None and (
            not structure_scale_weights
            or any(not math.isfinite(w) or w < 0 for w in structure_scale_weights)
            or sum(structure_scale_weights) <= 0
        ):
            raise ValueError("structure_scale_weights must be finite, nonnegative, with positive sum")
        self.edge_scale = edge_scale
        self.orientation_threshold = orientation_threshold
        self.structure_scale_weights = structure_scale_weights
        self.structure_known_weight = structure_known_weight
        self.perc = VGGPerceptualLoss(pretrained=vgg_pretrained) if use_perceptual else None
        if stage_weights is None:
            stage_weights = [(i + 1) / sum(range(1, K + 1)) for i in range(K)]
        if len(stage_weights) != K:
            raise ValueError(f"stage_weights length must equal K={K}, got {len(stage_weights)}")
        self.stage_weights = stage_weights

    @classmethod
    def from_config(cls, cfg: Mapping) -> "InpaintingLoss":
        """Construct the same loss for train/evaluate from a full project config."""
        loss = cfg.get("loss", {})
        return cls(
            K=cfg.get("model", {}).get("K", 3),
            lambda_rec=loss.get("lambda_rec", 1.0),
            lambda_perc=loss.get("lambda_perc", 0.0),
            lambda_edge=loss.get("lambda_edge", 1.0),
            lambda_stage=loss.get("lambda_stage", 0.5),
            hole_weight=loss.get("hole_weight", 6.0),
            edge_hole_only=loss.get("edge_hole_only", True),
            use_perceptual=loss.get("use_perceptual", False),
            vgg_pretrained=loss.get("vgg_pretrained", False),
            stage_weights=loss.get("stage_weights"),
            lambda_p_cons=loss.get("lambda_p_cons", 0.0),
            lambda_n_m=loss.get("lambda_n_m", 0.0),
            lambda_struct=loss.get("lambda_struct", 0.0),
            lambda_structure_grad=loss.get("lambda_structure_grad", 0.0),
            lambda_structure_edge=loss.get("lambda_structure_edge", 0.0),
            lambda_structure_orientation=loss.get("lambda_structure_orientation", 0.0),
            lambda_structure_consistency=loss.get("lambda_structure_consistency", 0.0),
            edge_scale=loss.get("edge_scale", 0.1),
            orientation_threshold=loss.get("orientation_threshold", 0.05),
            structure_scale_weights=loss.get("structure_scale_weights"),
            structure_known_weight=loss.get("structure_known_weight", 0.1),
        )

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
            edge_gt = self._edge_target(gt, self.edge_scale)
            l_struct = self.masked_mean((edge_pred - edge_gt).abs(), 1.0 - M)

        structure_terms = {name: pred.new_zeros(()) for name in self.structure_loss_weights}
        structure = outputs.get("structure")
        if structure is not None and any(w > 0 for w in self.structure_loss_weights.values()):
            structure_terms = structure_losses(
                structure, gt, M, edge_scale=self.edge_scale,
                orientation_threshold=self.orientation_threshold,
                scale_weights=self.structure_scale_weights, known_weight=self.structure_known_weight,
            )

        total = (
            self.lambda_rec * l_rec
            + self.lambda_perc * l_perc
            + self.lambda_edge * l_edge
            + self.lambda_stage * l_stage
            + self.lambda_p_cons * l_p_cons
            + self.lambda_n_m * l_n_m
            + self.lambda_struct * l_struct
            + sum(self.structure_loss_weights[name] * value for name, value in structure_terms.items())
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
            **{name: value.detach() for name, value in structure_terms.items()},
            # pred already is the final readout; expose its existing reconstruction
            # loss for logging without adding it to total a second time.
            "readout": l_rec.detach() if outputs.get("readout") is not None else pred.new_zeros(()),
        }

    @staticmethod
    def _edge_target(gt: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
        return edge_strength(grad(gt), scale=scale)
