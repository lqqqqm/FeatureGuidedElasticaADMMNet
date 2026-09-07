from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.diagnostics import checkpoint_coupling_scale, assert_finite_outputs
from fg_elastica_inpaint.utils.image import pil_to_tensor, tensor_to_pil



def load_rgb(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    t = pil_to_tensor(img, value_range="minus_one_to_one")
    return t.unsqueeze(0)



def load_mask(path: str, size: int, invert: bool = False) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    m = np.asarray(mask, dtype=np.float32) / 255.0
    m = (m > 0.5).astype(np.float32)
    if invert:
        m = 1.0 - m
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0)



def save_tensor_as_image(t: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(t).save(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True, help="Mask with white=known and black=hole by default.")
    parser.add_argument("--output_dir", type=str, default="./infer_out")
    parser.add_argument("--invert_mask", action="store_true", help="Use when your mask is white=hole, black=known.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = cfg["data"].get("image_size", 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    model = FeatureGuidedElasticaADMMNet.from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gt = load_rgb(args.image, size).to(device)
    M = load_mask(args.mask, size, invert=args.invert_mask).to(device)
    I_m = gt * M

    outputs = model(I_m, M, rho_scale=checkpoint_coupling_scale(cfg, ckpt))
    assert_finite_outputs(outputs)
    pred = outputs["pred"]
    comp = outputs["comp"]

    out_dir = Path(args.output_dir)
    save_tensor_as_image(I_m, out_dir / "masked.png")
    save_tensor_as_image(pred, out_dir / "pred.png")
    save_tensor_as_image(comp, out_dir / "composite.png")
    save_tensor_as_image(gt, out_dir / "input_resized.png")
    print(f"Saved outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
