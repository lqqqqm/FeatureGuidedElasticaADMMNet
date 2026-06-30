from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from fg_elastica_inpaint.data.dataset import build_dataloaders
from fg_elastica_inpaint.losses import InpaintingLoss
from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.metrics import (
    MASK_RATIO_BUCKETS,
    FrechetInceptionDistance,
    OptionalLPIPS,
    composite_completed,
    evaluate_per_image,
    mask_ratio_bucket,
    summarize_bucket_metrics,
    summarize_metric_items,
    validation_sanity,
)
from fg_elastica_inpaint.utils.misc import AverageMeter, append_csv_row

LOSS_KEYS = ["total", "rec", "edge", "perc", "stage", "p_cons", "n_m", "struct"]
HOLE_METRIC_COLUMNS = {"psnr_hole", "ssim_hole", "lpips_hole", "edge_f1_hole", "gradient_l1_hole"}


def progress_bar_enabled(cfg: dict) -> bool:
    """Use progress bars only in interactive terminals to avoid noisy batch logs."""
    return bool(cfg.get("logging", {}).get("progress_bar", sys.stderr.isatty()))


def remove_hole_metric_columns(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [name for name in (reader.fieldnames or []) if name not in HOLE_METRIC_COLUMNS and not name.endswith("_hole")]
        rows = [{key: value for key, value in row.items() if key in fieldnames} for row in reader]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        "ssim",
        "l1",
        "edge_f1",
        "gradient_l1",
        "boundary_consistency",
        "lpips",
    ]
    metric_meters = {name: AverageMeter() for name in metric_keys}
    lpips_metric = OptionalLPIPS(device) if cfg.get("eval", {}).get("compute_lpips", True) else None
    fid_metric = FrechetInceptionDistance(device) if cfg.get("eval", {}).get("compute_fid", False) else None
    fid_bucket_metrics = (
        {bucket: FrechetInceptionDistance(device) for bucket, _, _ in MASK_RATIO_BUCKETS}
        if fid_metric is not None
        else {}
    )
    per_image_items = []
    completed_min = float("inf")
    completed_max = float("-inf")

    for batch in tqdm(loader, desc=f"evaluate:{args.split}", disable=not progress_bar_enabled(cfg)):
        gt = batch["gt"].to(device)
        M = batch["mask"].to(device)
        I_m = batch["masked"].to(device)
        outputs = model(I_m, M)
        loss_dict = criterion(outputs, gt, M)
        pred = outputs["pred"]
        bs = gt.shape[0]
        for name, meter in loss_meters.items():
            meter.update(float(loss_dict[name]), bs)
        batch_items = evaluate_per_image(pred, gt, M, lpips_metric)
        per_image_items.extend(batch_items)
        metrics = summarize_metric_items(batch_items)
        for name, value in metrics.items():
            metric_meters[name].update(value, bs)
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
    if fid_metric is not None:
        try:
            results["fid"] = fid_metric.compute()
        except Exception as exc:
            results["fid"] = float("nan")
            print(f"[Sanity:{args.split}] WARNING: FID failed: {exc}")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    out_dir = Path(args.checkpoint).resolve().parent
    epoch = int(ckpt.get("epoch", -1))
    remove_hole_metric_columns(out_dir / "bucket_metrics.csv")
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


if __name__ == "__main__":
    main()
