"""Run local inpainting demos with reproducible random masks."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fg_elastica_inpaint.data.mask_generator import RandomMaskGenerator
from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.image import pil_to_tensor, tensor_to_pil


CONFIG_PATH = ROOT / "output" / "config_resolved_vgg.yaml"
CHECKPOINT_PATH = ROOT / "output" / "best_vgg.pt"
OUTPUT_DIR = ROOT / "output" / "infer_vgg_random_masks"
IMAGE_PATHS = [ROOT / "test_data" / "1.jpg", ROOT / "test_data" / "2.jpg"]
MASK_SETTINGS = [
    ("small_10_20", (0.10, 0.20), "thin_distributed"),
    ("medium_30_45", (0.30, 0.45), "large_contiguous"),
    ("large_45_60", (0.45, 0.60), "large_contiguous"),
]


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

    rows = ["image,mask_setting,hole_ratio,device"]
    with torch.no_grad():
        for image_index, image_path in enumerate(IMAGE_PATHS, start=1):
            image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            gt = pil_to_tensor(image, value_range="minus_one_to_one").unsqueeze(0).to(device)
            for setting_index, (name, bucket, topology) in enumerate(MASK_SETTINGS, start=1):
                random.seed(4200 + image_index * 10 + setting_index)
                np.random.seed(4200 + image_index * 10 + setting_index)
                generator = RandomMaskGenerator(
                    size=size,
                    hole_buckets=[bucket],
                    bucket_probs=[1.0],
                    topology_modes=[topology],
                    topology_probs=[1.0],
                )
                mask = None
                for _ in range(100):
                    candidate = generator(size, size)
                    ratio = float(1.0 - candidate.mean())
                    if bucket[0] <= ratio <= bucket[1]:
                        mask = candidate.unsqueeze(0).to(device)
                        break
                if mask is None:
                    raise RuntimeError(f"Could not generate a {bucket[0]:.0%}-{bucket[1]:.0%} mask for {name}.")
                masked = gt * mask
                outputs = model(masked, mask)

                sample_dir = OUTPUT_DIR / f"image_{image_index}" / name
                sample_dir.mkdir(parents=True, exist_ok=True)
                tensor_to_pil(gt.cpu()).save(sample_dir / "input.png")
                Image.fromarray((mask[0, 0].cpu().numpy() * 255).astype(np.uint8), mode="L").save(sample_dir / "mask.png")
                tensor_to_pil(masked.cpu()).save(sample_dir / "masked.png")
                tensor_to_pil(outputs["pred"].cpu()).save(sample_dir / "pred.png")
                tensor_to_pil(outputs["comp"].cpu()).save(sample_dir / "composite.png")
                hole_ratio = float(1.0 - mask.mean().cpu())
                rows.append(f"{image_path.name},{name},{hole_ratio:.4f},{device.type}")
                print(f"{image_path.name} | {name} | hole={hole_ratio:.1%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
