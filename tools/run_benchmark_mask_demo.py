"""Evaluate a checkpoint on the fixed masks saved by the CPU mask benchmarks."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.image import pil_to_tensor, tensor_to_pil
from fg_elastica_inpaint.utils.metrics import evaluate_batch


CONFIG_PATH = ROOT / "output" / "config_resolved_vgg.yaml"
CHECKPOINT_PATH = ROOT / "output" / "best_vgg.pt"
IMAGE_DIR = ROOT / "test_data"
MASK_ROOTS = [
    ROOT / "output" / "cpu_mask_benchmark_scattered",
    ROOT / "output" / "cpu_mask_benchmark_scattered_varied",
]
OUTPUT_DIR = ROOT / "output" / "infer_vgg_benchmark_masks"


def load_mask(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    values = (np.asarray(image, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    size = int(cfg["data"].get("image_size", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    model = FeatureGuidedElasticaADMMNet.from_config(cfg).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, str | float]] = []
    with torch.no_grad():
        for mask_root in MASK_ROOTS:
            benchmark = mask_root.name
            mask_paths = sorted(mask_root.glob("*/*/mask.png"))
            if not mask_paths:
                raise FileNotFoundError(f"No masks found in {mask_root}")
            for mask_path in mask_paths:
                image_id = mask_path.parent.parent.name
                variant = mask_path.parent.name
                image_path = IMAGE_DIR / f"{image_id}.jpg"
                if not image_path.exists():
                    raise FileNotFoundError(f"No test image matches {mask_path}: {image_path}")

                image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
                gt = pil_to_tensor(image, value_range="minus_one_to_one").unsqueeze(0).to(device)
                mask = load_mask(mask_path, size).to(device)
                masked = gt * mask
                outputs = model(masked, mask)
                pred = outputs["pred"]
                composite = outputs["comp"]
                metrics = evaluate_batch(pred, gt, mask)

                sample_dir = OUTPUT_DIR / benchmark / image_id / variant
                sample_dir.mkdir(parents=True, exist_ok=True)
                tensor_to_pil(gt.cpu()).save(sample_dir / "input.png")
                Image.fromarray((mask[0, 0].cpu().numpy() * 255).astype(np.uint8), mode="L").save(sample_dir / "mask.png")
                tensor_to_pil(masked.cpu()).save(sample_dir / "masked.png")
                tensor_to_pil(pred.cpu()).save(sample_dir / "pred.png")
                tensor_to_pil(composite.cpu()).save(sample_dir / "composite.png")

                row: dict[str, str | float] = {
                    "benchmark": benchmark,
                    "image": image_path.name,
                    "variant": variant,
                    "hole_ratio": float(1.0 - mask.mean().cpu()),
                    "psnr": metrics["psnr"],
                    "psnr_hole": metrics["psnr_hole"],
                    "ssim": metrics["ssim"],
                    "ssim_hole": metrics["ssim_hole"],
                }
                rows.append(row)
                print(
                    f"{benchmark} | {image_path.name} | {variant} | "
                    f"hole={row['hole_ratio']:.1%} | PSNR-hole={row['psnr_hole']:.2f} | "
                    f"SSIM-hole={row['ssim_hole']:.4f}"
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["benchmark", "image", "variant", "hole_ratio", "psnr", "psnr_hole", "ssim", "ssim_hole"]
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
