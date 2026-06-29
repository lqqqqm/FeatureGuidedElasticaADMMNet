from __future__ import annotations

import torch
import torch.nn as nn


class CorrectionHead(nn.Module):
    def __init__(
        self,
        variable_channels: int,
        feature_channels: int,
        cout: int,
        hidden_channels: int = 32,
        gate_init: float = 0.1,
        zero_init_last: bool = True,
    ):
        super().__init__()
        cin = variable_channels + feature_channels + cout + 1
        self.trunk = nn.Sequential(
            nn.Conv2d(cin, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.delta = nn.Conv2d(hidden_channels, cout, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(hidden_channels, cout, kernel_size=3, padding=1)
        if zero_init_last:
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.gate.bias, torch.logit(torch.tensor(gate_init)).item())

    def forward(
        self,
        variable: torch.Tensor,
        adapted_feature: torch.Tensor,
        mask: torch.Tensor,
        base_correction: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat([variable, adapted_feature, mask, base_correction], dim=1)
        feature = self.trunk(x)
        delta_res = self.delta(feature)
        gate = torch.sigmoid(self.gate(feature))
        correction = gate * (base_correction + delta_res)
        if return_aux:
            return correction, {"delta_res": delta_res, "gate": gate}
        return correction
