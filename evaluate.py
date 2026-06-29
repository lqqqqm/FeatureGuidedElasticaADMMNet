from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from fg_elastica_inpaint.data.dataset import build_dataloaders
from fg_elastica_inpaint.losses import InpaintingLoss
from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.metrics import OptionalLPIPS, evaluate_batch
from fg_elastica_inpaint.utils.misc import AverageMeter

LOSS_KEYS = ["total", "rec", "edge", "perc", "stage", "p_cons", "n_m", "struct"]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    _, val_loader, test_loader = build_dataloaders(cfg)
    loader = test_loader if args.split == "test" else val_loader
    if loader is None:
        raise ValueError(f"No dataloader available for split={args.split}")

    model = FeatureGuidedElasticaADMMNet.from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loss_cfg = cfg["loss"]
    criterion = InpaintingLoss(
        K=cfg["model"].get("K", 3),
        lambda_rec=loss_cfg.get("lambda_rec", 1.0),
        lambda_perc=loss_cfg.get("lambda_perc", 0.0),
        lambda_edge=loss_cfg.get("lambda_edge", 1.0),
        lambda_stage=loss_cfg.get("lambda_stage", 0.5),
        hole_weight=loss_cfg.get("hole_weight", 6.0),
        edge_hole_only=loss_cfg.get("edge_hole_only", True),
        use_perceptual=loss_cfg.get("use_perceptual", False),
        vgg_pretrained=loss_cfg.get("vgg_pretrained", False),
        stage_weights=loss_cfg.get("stage_weights"),
        lambda_p_cons=loss_cfg.get("lambda_p_cons", 0.0),
        lambda_n_m=loss_cfg.get("lambda_n_m", 0.0),
        lambda_struct=loss_cfg.get("lambda_struct", 0.0),
    ).to(device)

    loss_meters = {name: AverageMeter() for name in LOSS_KEYS}
    metric_keys = [
        "psnr",
        "psnr_hole",
        "ssim",
        "ssim_hole",
        "edge_f1",
        "gradient_l1",
        "boundary_consistency",
        "lpips",
        "lpips_hole",
    ]
    metric_meters = {name: AverageMeter() for name in metric_keys}
    lpips_metric = OptionalLPIPS(device) if cfg.get("eval", {}).get("compute_lpips", True) else None

    for batch in tqdm(loader, desc=f"evaluate:{args.split}"):
        gt = batch["gt"].to(device)
        M = batch["mask"].to(device)
        I_m = batch["masked"].to(device)
        outputs = model(I_m, M)
        loss_dict = criterion(outputs, gt, M)
        pred = outputs["pred"]
        bs = gt.shape[0]
        for name, meter in loss_meters.items():
            meter.update(float(loss_dict[name]), bs)
        metrics = evaluate_batch(pred, gt, M, lpips_metric)
        for name, value in metrics.items():
            metric_meters[name].update(value, bs)

    results = {name: meter.avg for name, meter in loss_meters.items()}
    results.update({name: meter.avg for name, meter in metric_meters.items() if meter.count > 0})
    print(json.dumps(results, indent=2, ensure_ascii=False))

    out_dir = Path(args.checkpoint).resolve().parent
    with (out_dir / f"eval_{args.split}.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
