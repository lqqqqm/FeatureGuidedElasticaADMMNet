from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR



def build_warmup_cosine_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.05) -> LambdaLR:
    total_steps = max(total_steps, 1)
    warmup_steps = max(warmup_steps, 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
