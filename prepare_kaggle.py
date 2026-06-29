"""Prepare deterministic CelebA-HQ splits and fixed masks for Kaggle training.

Run after unzipping the project to /kaggle/working.  Source images stay in
/kaggle/input; only text lists and masks are created under /kaggle/working.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from fg_elastica_inpaint.data.mask_generator import RandomMaskGenerator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_list(path: Path, items: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(item) for item in items) + "\n", encoding="utf-8")


def write_masks(directory: Path, list_path: Path, count: int, generator: RandomMaskGenerator) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        # The project convention is white=known (1), black=hole (0).
        mask = generator(256, 256).squeeze(0).numpy()
        path = directory / f"mask_{index:05d}.png"
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)
        paths.append(path)
    write_list(list_path, paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CelebA-HQ lists and fixed masks for the Kaggle experiment.")
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="CelebA-HQ image directory mounted under /kaggle/input.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--max-images", type=int, default=10_000, help="Use this many images after seeded shuffling; 0 uses all.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.images_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.images_root}")
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Validation and test ratios must both be positive and sum to less than 1.")

    images = sorted(
        (path.resolve() for path in args.images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )
    if args.max_images > 0:
        if len(images) < args.max_images:
            raise ValueError(f"Requested {args.max_images} images but found only {len(images)} under {args.images_root}")
    elif args.max_images < 0:
        raise ValueError("--max-images must be zero or a positive integer.")

    random.Random(args.seed).shuffle(images)
    if args.max_images:
        images = images[: args.max_images]
    val_count = round(len(images) * args.val_ratio)
    test_count = round(len(images) * args.test_ratio)
    train_count = len(images) - val_count - test_count
    if train_count < 1 or val_count < 1 or test_count < 1:
        raise ValueError("The selected image count is too small for these split ratios.")
    train = images[:train_count]
    val = images[train_count : train_count + val_count]
    test = images[train_count + val_count :]

    split_dir = args.work_dir / "data_splits"
    write_list(split_dir / "train.txt", train)
    write_list(split_dir / "val.txt", val)
    write_list(split_dir / "test.txt", test)

    random.seed(args.seed)
    np.random.seed(args.seed)
    generator = RandomMaskGenerator(
        size=256,
        hole_buckets=[(0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.6)],
        bucket_probs=[0.2, 0.2, 0.25, 0.35],
        topology_modes=["thin_distributed", "scattered_irregular", "large_contiguous", "mixed"],
        topology_probs=[0.3, 0.3, 0.25, 0.15],
    )
    mask_dir = args.work_dir / "fixed_masks"
    write_masks(mask_dir / "val", mask_dir / "val_masks.txt", len(val), generator)
    write_masks(mask_dir / "test", mask_dir / "test_masks.txt", len(test), generator)

    print(f"Prepared {len(images)} images: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Split lists: {split_dir}")
    print(f"Fixed masks: {mask_dir}")
    print("Next: python train.py --config configs/mvp_kaggle_9k.yaml")


if __name__ == "__main__":
    main()
