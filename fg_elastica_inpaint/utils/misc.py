from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class MaximumMeter:
    """Keep the worst observed diagnostic across batches."""
    def __init__(self):
        self.value = -float("inf")
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = max(self.value, value)
        self.count += n

    @property
    def avg(self) -> float:
        return self.value if self.count else 0.0



def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def append_csv_row(path: str | Path, row: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_keys = list(row.keys())
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row_keys)
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        existing_keys = list(reader.fieldnames or [])

    fieldnames = list(existing_keys)
    for key in row_keys:
        if key not in fieldnames:
            fieldnames.append(key)

    if fieldnames != existing_keys:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow(existing)
            writer.writerow(row)
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)



def save_checkpoint(state: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
