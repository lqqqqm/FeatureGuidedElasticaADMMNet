"""Prepare image lists, create a resolved config, and optionally run an experiment.

Example (PowerShell):
    python tools/prepare_and_run.py --images D:/datasets/places2 --run
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(root: Path) -> list[Path]:
    """Return supported image files in a deterministic order."""
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    return sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )


def write_list(path: Path, items: Iterable[Path]) -> int:
    values = [str(item) for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")
    return len(values)


def split_files(files: list[Path], val_ratio: float, test_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("--val-ratio and --test-ratio must be non-negative and add up to less than 1.")
    if len(files) < 3:
        raise ValueError("At least 3 images are required to create train/val/test splits.")

    shuffled = files.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    test_count = max(1, round(len(shuffled) * test_ratio))
    train_count = len(shuffled) - val_count - test_count
    if train_count < 1:
        raise ValueError("Split ratios leave no image for training.")
    return shuffled[:train_count], shuffled[train_count : train_count + val_count], shuffled[train_count + val_count :]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reproducible train/val/test file lists and optionally train and evaluate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", required=True, help="Root directory containing all source images (searched recursively).")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Base training YAML configuration.")
    parser.add_argument("--prepared-dir", default="prepared_data", help="Where split lists and the resolved config are stored.")
    parser.add_argument("--val-masks", help="Optional directory of fixed masks for validation.")
    parser.add_argument("--test-masks", help="Optional directory of fixed masks for testing.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", help="Override config run_name; useful for distinguishing experiments.")
    parser.add_argument("--run", action="store_true", help="After preparation, run training and then evaluate best.pt on test.")
    parser.add_argument("--skip-eval", action="store_true", help="With --run, train only and do not invoke evaluate.py.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_images = image_files(Path(args.images).expanduser())
    train, val, test = split_files(source_images, args.val_ratio, args.test_ratio, args.seed)

    prepared_dir = Path(args.prepared_dir).expanduser()
    if not prepared_dir.is_absolute():
        prepared_dir = project_root / prepared_dir
    prepared_dir = prepared_dir.resolve()
    train_list = prepared_dir / "train.txt"
    val_list = prepared_dir / "val.txt"
    test_list = prepared_dir / "test.txt"
    counts = {
        "train": write_list(train_list, train),
        "val": write_list(val_list, val),
        "test": write_list(test_list, test),
    }

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.setdefault("data", {})
    config["seed"] = args.seed
    config["data"].update({
        "train_list": str(train_list),
        "val_list": str(val_list),
        "test_list": str(test_list),
    })
    for split, mask_dir in (("val", args.val_masks), ("test", args.test_masks)):
        key = f"{split}_mask_list"
        if mask_dir:
            masks = image_files(Path(mask_dir).expanduser())
            if not masks:
                raise ValueError(f"No supported mask images found in: {mask_dir}")
            config["data"][key] = str(prepared_dir / f"{split}_masks.txt")
            counts[f"{split}_masks"] = write_list(Path(config["data"][key]), masks)
        else:
            config["data"][key] = None
    if args.run_name:
        config["run_name"] = args.run_name

    resolved_config = prepared_dir / "experiment.yaml"
    with resolved_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    print(f"Prepared {len(source_images)} images in {prepared_dir}")
    print(" | ".join(f"{name}={count}" for name, count in counts.items()))
    print(f"Experiment config: {resolved_config}")
    if not args.run:
        print(f"Next: {sys.executable} train.py --config {resolved_config}")
        return

    subprocess.run([sys.executable, "train.py", "--config", str(resolved_config)], cwd=project_root, check=True)
    if args.skip_eval:
        return
    checkpoint = project_root / config.get("output_dir", "./outputs") / config["run_name"] / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training finished but best checkpoint was not found: {checkpoint}")
    subprocess.run(
        [sys.executable, "evaluate.py", "--config", str(resolved_config), "--checkpoint", str(checkpoint), "--split", "test"],
        cwd=project_root,
        check=True,
    )


if __name__ == "__main__":
    main()
