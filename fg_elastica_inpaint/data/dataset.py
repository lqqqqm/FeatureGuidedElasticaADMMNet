from __future__ import annotations

import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageEnhance
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.image import pil_to_tensor
from .mask_generator import FixedMaskLoader, RandomMaskGenerator



def _resize_short_side(img: Image.Image, short_side: int) -> Image.Image:
    w, h = img.size
    if min(w, h) == short_side:
        return img
    if w < h:
        new_w = short_side
        new_h = int(round(h * short_side / w))
    else:
        new_h = short_side
        new_w = int(round(w * short_side / h))
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)



def _random_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w == size and h == size:
        return img
    left = random.randint(0, max(w - size, 0))
    top = random.randint(0, max(h - size, 0))
    return img.crop((left, top, left + size, top + size))



def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = max((w - size) // 2, 0)
    top = max((h - size) // 2, 0)
    return img.crop((left, top, left + size, top + size))



def _random_hflip(img: Image.Image, p: float = 0.5) -> Image.Image:
    if random.random() < p:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img



def _apply_color_jitter(img: Image.Image, brightness: float = 0.1, contrast: float = 0.1, saturation: float = 0.1) -> Image.Image:
    factors = {
        ImageEnhance.Brightness: 1.0 + random.uniform(-brightness, brightness),
        ImageEnhance.Contrast: 1.0 + random.uniform(-contrast, contrast),
        ImageEnhance.Color: 1.0 + random.uniform(-saturation, saturation),
    }
    enhancers = list(factors.items())
    random.shuffle(enhancers)
    for enhancer_cls, factor in enhancers:
        img = enhancer_cls(img).enhance(factor)
    return img


class InpaintingImageDataset(Dataset):
    def __init__(
        self,
        file_list: Sequence[str],
        image_size: int = 256,
        train: bool = True,
        resize_short_to: int = 286,
        fixed_mask_paths: Optional[Sequence[str]] = None,
        hole_buckets=None,
        bucket_probs=None,
        mask_topology_modes=None,
        mask_topology_probs=None,
        eval_mask_seed: int = 1701,
    ):
        self.paths = [str(Path(p).expanduser()) for p in file_list]
        self.image_size = image_size
        self.train = train
        self.resize_short_to = resize_short_to
        self.eval_mask_seed = eval_mask_seed
        self.fixed_masks = FixedMaskLoader(fixed_mask_paths, image_size=image_size) if fixed_mask_paths else None
        self.random_masks = RandomMaskGenerator(
            size=image_size,
            hole_buckets=hole_buckets,
            bucket_probs=bucket_probs,
            topology_modes=mask_topology_modes,
            topology_probs=mask_topology_probs,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _transform(self, img: Image.Image) -> Image.Image:
        img = _resize_short_side(img, self.resize_short_to)
        if self.train:
            img = _random_crop(img, self.image_size)
            img = _random_hflip(img, p=0.5)
            img = _apply_color_jitter(img)
        else:
            img = _center_crop(img, self.image_size)
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self._transform(img)
        gt = pil_to_tensor(img, value_range="minus_one_to_one")
        if self.fixed_masks is not None:
            M = self.fixed_masks[idx]
        elif self.train:
            M = self.random_masks(self.image_size, self.image_size)
        else:
            # Stable index -> mask mapping even without an external mask list.
            py_state, np_state = random.getstate(), np.random.get_state()
            try:
                random.seed(self.eval_mask_seed + idx)
                np.random.seed((self.eval_mask_seed + idx) % 2**32)
                M = self.random_masks(self.image_size, self.image_size)
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
        I_m = gt * M
        return {
            "gt": gt,
            "mask": M,
            "masked": I_m,
            "path": path,
        }



def read_file_list(path: str | Path, limit: Optional[int] = None) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File list not found: {p}")
    items = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        items = items[:limit]
    return items



def build_dataloader(
    file_list_path: str | Path,
    image_size: int,
    batch_size: int,
    train: bool,
    num_workers: int,
    fixed_mask_list: Optional[str | Path] = None,
    resize_short_to: int = 286,
    hole_buckets=None,
    bucket_probs=None,
    mask_topology_modes=None,
    mask_topology_probs=None,
    limit: Optional[int] = None,
    eval_mask_seed: int = 1701,
) -> DataLoader:
    file_list = read_file_list(file_list_path, limit=limit)
    fixed_mask_paths = read_file_list(fixed_mask_list) if fixed_mask_list else None
    ds = InpaintingImageDataset(
        file_list=file_list,
        image_size=image_size,
        train=train,
        resize_short_to=resize_short_to,
        fixed_mask_paths=fixed_mask_paths,
        hole_buckets=hole_buckets,
        bucket_probs=bucket_probs,
        mask_topology_modes=mask_topology_modes,
        mask_topology_probs=mask_topology_probs,
        eval_mask_seed=eval_mask_seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )



def build_dataloaders(cfg: Dict):
    data_cfg = cfg["data"]
    train_loader = build_dataloader(
        file_list_path=data_cfg["train_list"],
        image_size=data_cfg.get("image_size", 256),
        batch_size=data_cfg.get("batch_size", 4),
        train=True,
        num_workers=data_cfg.get("num_workers", 4),
        resize_short_to=data_cfg.get("resize_short_to", 286),
        hole_buckets=data_cfg.get("hole_buckets"),
        bucket_probs=data_cfg.get("bucket_probs"),
        mask_topology_modes=data_cfg.get("mask_topology_modes"),
        mask_topology_probs=data_cfg.get("mask_topology_probs"),
        limit=data_cfg.get("train_limit"),
    )
    val_loader = None
    test_loader = None
    if data_cfg.get("val_list"):
        val_loader = build_dataloader(
            file_list_path=data_cfg["val_list"],
            image_size=data_cfg.get("image_size", 256),
            batch_size=data_cfg.get("val_batch_size", data_cfg.get("batch_size", 4)),
            train=False,
            num_workers=data_cfg.get("num_workers", 4),
            fixed_mask_list=data_cfg.get("val_mask_list"),
            resize_short_to=data_cfg.get("resize_short_to", 286),
            limit=data_cfg.get("val_limit"),
            hole_buckets=data_cfg.get("hole_buckets"),
            bucket_probs=data_cfg.get("bucket_probs"),
            mask_topology_modes=data_cfg.get("mask_topology_modes"),
            mask_topology_probs=data_cfg.get("mask_topology_probs"),
            eval_mask_seed=data_cfg.get("eval_mask_seed", 1701),
        )
    if data_cfg.get("test_list"):
        test_loader = build_dataloader(
            file_list_path=data_cfg["test_list"],
            image_size=data_cfg.get("image_size", 256),
            batch_size=data_cfg.get("test_batch_size", data_cfg.get("batch_size", 4)),
            train=False,
            num_workers=data_cfg.get("num_workers", 4),
            fixed_mask_list=data_cfg.get("test_mask_list"),
            resize_short_to=data_cfg.get("resize_short_to", 286),
            limit=data_cfg.get("test_limit"),
            hole_buckets=data_cfg.get("hole_buckets"),
            bucket_probs=data_cfg.get("bucket_probs"),
            mask_topology_modes=data_cfg.get("mask_topology_modes"),
            mask_topology_probs=data_cfg.get("mask_topology_probs"),
            eval_mask_seed=data_cfg.get("test_mask_seed", 2701),
        )
    return train_loader, val_loader, test_loader
