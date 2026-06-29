from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    def __init__(self, pretrained: bool = False, layers: Iterable[str] | None = None):
        super().__init__()
        layers = list(layers or ["relu1_2", "relu2_2", "relu3_4"])
        self.target_layers = set(layers)
        try:
            from torchvision.models import VGG19_Weights, vgg19  # lazy import to avoid hard dependency during MVP runs.
        except Exception as exc:
            raise RuntimeError(
                "torchvision is required only when use_perceptual=true. In this environment it could not be imported."
            ) from exc

        try:
            weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            features = vgg19(weights=weights).features
        except Exception as exc:
            raise RuntimeError(
                "Failed to build VGG19 perceptual network. If running offline, set loss.use_perceptual=false or"
                " set loss.vgg_pretrained=false after caching the weights manually."
            ) from exc

        self.blocks = nn.ModuleList()
        ranges = {
            "relu1_2": (0, 4),
            "relu2_2": (4, 9),
            "relu3_4": (9, 18),
        }
        order = ["relu1_2", "relu2_2", "relu3_4"]
        prev = 0
        for name in order:
            end = ranges[name][1]
            block = nn.Sequential(*[features[i] for i in range(prev, end)])
            self.blocks.append(block)
            prev = end
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad_(False)
        self.blocks.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.order = order

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = self._prep(pred)
        y = self._prep(target)
        loss = x.new_tensor(0.0)
        for name, block in zip(self.order, self.blocks):
            x = block(x)
            y = block(y)
            if name in self.target_layers:
                loss = loss + F.l1_loss(x, y)
        return loss
