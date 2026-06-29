from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image
import torch



def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)



def pil_to_tensor(img: Image.Image, value_range: str = "minus_one_to_one") -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.transpose(2, 0, 1) / 255.0
    t = torch.from_numpy(arr)
    if value_range == "minus_one_to_one":
        t = t * 2.0 - 1.0
    return t



def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = denorm(t.detach().cpu())
    if t.ndim == 4:
        t = t[0]
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")



def _make_grid(batch: torch.Tensor, nrow: int = 4) -> torch.Tensor:
    batch = denorm(batch.detach().cpu())
    B, C, H, W = batch.shape
    ncol = nrow
    nrows = (B + ncol - 1) // ncol
    grid = torch.zeros(C, nrows * H, ncol * W, dtype=batch.dtype)
    for idx in range(B):
        r = idx // ncol
        c = idx % ncol
        grid[:, r * H : (r + 1) * H, c * W : (c + 1) * W] = batch[idx]
    return grid



def save_triplet(masked: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = torch.cat([masked, pred, gt], dim=-1)
    tensor_to_pil(vis).save(path)



def save_batch_grid(batch: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = _make_grid(batch, nrow=nrow)
    tensor_to_pil(grid).save(path)
