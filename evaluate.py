from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fg_elastica_inpaint.data.dataset import build_dataloaders
from fg_elastica_inpaint.losses import InpaintingLoss
from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.diagnostics import (checkpoint_coupling_scale, output_diagnostics,
    evaluate_outputs, identified_rows, write_rows)
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.metrics import (
    MASK_RATIO_BUCKETS,
    FrechetInceptionDistance,
    OptionalLPIPS,
    composite_completed,
    mask_ratio_bucket,
    summarize_bucket_metrics,
    summarize_metric_items,
    validation_sanity,
)
from fg_elastica_inpaint.utils.misc import AverageMeter, MaximumMeter, append_csv_row
from fg_elastica_inpaint.utils.progress import progress_bar


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
    rho_scale = checkpoint_coupling_scale(cfg, ckpt)

    criterion = InpaintingLoss.from_config(cfg).to(device)

    loss_meters = {}
    metric_meters = {}
    lpips_metric = OptionalLPIPS(device) if cfg.get("eval", {}).get("compute_lpips", True) else None
    fid_metric = FrechetInceptionDistance(device) if cfg.get("eval", {}).get("compute_fid", False) else None
    fid_bucket_metrics = (
        {bucket: FrechetInceptionDistance(device) for bucket, _, _ in MASK_RATIO_BUCKETS}
        if fid_metric is not None
        else {}
    )
    per_image_items = []
    named_items = []
    completed_min = float("inf")
    completed_max = float("-inf")

    pbar = progress_bar(loader, desc=f"Evaluate {args.split}", cfg=cfg)
    for batch in pbar:
        gt = batch["gt"].to(device)
        M = batch["mask"].to(device)
        I_m = batch["masked"].to(device)
        outputs = model(I_m, M, rho_scale=rho_scale)
        loss_dict = criterion(outputs, gt, M)
        pred = outputs["pred"]
        bs = gt.shape[0]
        for name, value in loss_dict.items():
            loss_meters.setdefault(name, AverageMeter()).update(float(value), bs)
        batch_items = evaluate_outputs(outputs, gt, M, cfg, lpips_metric)
        named_items.extend(identified_rows(batch_items, batch, len(per_image_items)))
        per_image_items.extend(batch_items)
        metrics = summarize_metric_items(batch_items)
        metrics.update(output_diagnostics(outputs))
        for name, value in metrics.items():
            metric_meters.setdefault(name, MaximumMeter() if name.endswith("_max") else AverageMeter()).update(value, bs)
        pbar.set_postfix({key: f"{metric_meters[key].avg:.3f}" for key in ("psnr", "ssim")
                         if key in metric_meters and metric_meters[key].count > 0}, refresh=False)
        completed = composite_completed(pred, gt, M)
        completed_min = min(completed_min, float(completed.min().detach()))
        completed_max = max(completed_max, float(completed.max().detach()))
        if fid_metric is not None:
            fid_metric.update(completed, gt)
            for sample_idx, item in enumerate(batch_items):
                bucket = mask_ratio_bucket(float(item["hole_ratio"]))
                fid_bucket_metrics[bucket].update(completed[sample_idx : sample_idx + 1], gt[sample_idx : sample_idx + 1])

    results = {name: meter.avg for name, meter in loss_meters.items()}
    results.update({name: meter.avg for name, meter in metric_meters.items() if meter.count > 0})
    results.update(summarize_metric_items(per_image_items))
    results["rho_scale"] = rho_scale
    if fid_metric is not None:
        try:
            results["fid"] = fid_metric.compute()
        except Exception as exc:
            results["fid"] = float("nan")
            print(f"[Sanity:{args.split}] WARNING: FID failed: {exc}")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    out_dir = Path(args.checkpoint).resolve().parent
    epoch = int(ckpt.get("epoch", -1))
    for bucket_row in summarize_bucket_metrics(per_image_items):
        if fid_bucket_metrics and int(bucket_row["num_samples"]) > 0:
            try:
                bucket_row["fid"] = fid_bucket_metrics[str(bucket_row["bucket"])].compute()
            except Exception as exc:
                bucket_row["fid"] = float("nan")
                print(f"[Sanity:{args.split}] WARNING: bucket {bucket_row['bucket']} FID failed: {exc}")
        append_csv_row(out_dir / "bucket_metrics.csv", {"epoch": epoch, "split": args.split, **bucket_row})
    for line in validation_sanity(per_image_items, completed_min, completed_max):
        print(f"[Sanity:{args.split}] {line}")

    with (out_dir / f"eval_{args.split}.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    write_rows(out_dir / f"eval_{args.split}_per_image.csv", named_items)


if __name__ == "__main__":
    main()
