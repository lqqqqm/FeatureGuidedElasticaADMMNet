from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import ConvBlock


class LearnedStructurePrior(nn.Module):
    """Predict signed RGB gradients once per image; never predict RGB intensities."""

    def __init__(self):
        super().__init__()
        self.project3 = nn.Conv2d(256, 64, 1)
        self.project2 = nn.Conv2d(128, 32, 1)
        self.project1 = nn.Conv2d(64, 32, 1)
        self.block3 = ConvBlock(65, 64)
        self.block2 = ConvBlock(97, 64)
        self.block1 = ConvBlock(97, 32)
        self.gradient3 = nn.Conv2d(64, 6, 1)
        self.gradient2 = nn.Conv2d(64, 6, 1)
        self.gradient1 = nn.Conv2d(32, 6, 3, padding=1)
        self.edge = nn.Conv2d(32, 1, 3, padding=1)
        # Small nonzero weights allow supervision to reach the decoder at once.
        for head in (self.gradient1, self.gradient2, self.gradient3):
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)

    def forward(self, f1: Tensor, f2: Tensor, f3: Tensor, M: Tensor) -> dict:
        resize = lambda x, size: F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        z3 = self.block3(torch.cat([self.project3(f3), resize(M, f3.shape[-2:])], 1))
        z2 = self.block2(torch.cat([resize(z3, f2.shape[-2:]), self.project2(f2), resize(M, f2.shape[-2:])], 1))
        z1 = self.block1(torch.cat([resize(z2, f1.shape[-2:]), self.project1(f1), M], 1))
        gradients = [2 * torch.tanh(head(z)) for head, z in
                     ((self.gradient1, z1), (self.gradient2, z2), (self.gradient3, z3))]
        logits = self.edge(z1)
        return {"gradient": gradients[0], "gradient_pyramid": gradients,
                "edge_logits": logits, "edge": torch.sigmoid(logits)}
