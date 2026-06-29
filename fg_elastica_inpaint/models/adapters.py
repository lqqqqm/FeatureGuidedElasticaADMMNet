from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(64 + 128 + 256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        size = f1.shape[-2:]
        f2_up = F.interpolate(f2, size=size, mode="bilinear", align_corners=False)
        f3_up = F.interpolate(f3, size=size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([f1, f2_up, f3_up], dim=1))


class AdapterHead(nn.Module):
    def __init__(
        self,
        semantic_channels: int,
        variable_channels: int,
        residual_channels: int,
        cout: int = 32,
        hidden_channels: int = 64,
    ):
        super().__init__()
        condition_channels = variable_channels + residual_channels + 1
        self.semantic = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.condition = nn.Sequential(
            nn.Conv2d(condition_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.gamma = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.out = nn.Conv2d(hidden_channels, cout, kernel_size=3, padding=1)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        semantic: torch.Tensor,
        variable: torch.Tensor,
        mask: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        semantic_feature = self.semantic(semantic)
        condition_feature = self.condition(torch.cat([variable, residual, mask], dim=1))
        gamma = self.gamma(condition_feature)
        beta = self.beta(condition_feature)
        gate = torch.sigmoid(self.gate(condition_feature))
        adapted = semantic_feature * (1.0 + gamma) + beta
        return self.out(gate * adapted)
