from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from tqdm import tqdm

from fg_elastica_inpaint.data.dataset import build_dataloaders
from fg_elastica_inpaint.losses import InpaintingLoss
from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config, save_config
from fg_elastica_inpaint.utils.image import save_triplet
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
from fg_elastica_inpaint.utils.misc import AverageMeter, append_csv_row, count_parameters, save_checkpoint, set_seed
from fg_elastica_inpaint.utils.scheduler import build_warmup_cosine_scheduler

LOSS_KEYS = ["total", "rec", "edge", "perc", "stage", "p_cons", "n_m", "struct"]
HOLE_METRIC_COLUMNS = {"psnr_hole", "ssim_hole", "lpips_hole", "edge_f1_hole", "gradient_l1_hole"}


def _format_stats(stats: Optional[Dict], keys) -> str:
    if not stats:
        return "n/a"
    parts = []
    for key in keys:
        if key in stats:
            parts.append(f"{key}={stats[key]:.4f}")
    return ", ".join(parts) if parts else "n/a"


def _remove_hole_metric_columns(path: Path) -> None:
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


def build_monitor(cfg: Dict, has_val: bool) -> Dict[str, object]:
    monitor_cfg = cfg.get("monitor", {})
    metric = monitor_cfg.get("metric")
    if metric is None:
        metric = "psnr" if has_val else "total"
    if metric == "psnr_hole":
        metric = "psnr"

    mode = monitor_cfg.get("mode")
    if mode is None:
        mode = "min" if metric in {"total", "rec", "edge", "perc", "stage"} else "max"
    if mode not in {"min", "max"}:
        raise ValueError(f"monitor.mode must be 'min' or 'max', got {mode}")

    return {
        "metric": str(metric),
        "mode": mode,
        "patience": int(monitor_cfg.get("patience", 12)),
        "min_delta": float(monitor_cfg.get("min_delta", 0.01)),
        "warmup_epochs": int(monitor_cfg.get("warmup_epochs", 0)),
        "best_score": None,
        "best_epoch": 0,
        "bad_epochs": 0,
        "source": "val" if has_val else "train",
    }


def monitor_value(stats: Optional[Dict], monitor: Dict[str, object]) -> Optional[float]:
    if stats is None:
        return None
    metric = str(monitor["metric"])
    value = stats.get(metric)
    return float(value) if value is not None else None


def monitor_improved(current: float, best: Optional[float], mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best + min_delta
    return current < best - min_delta


def build_output_dir(cfg: Dict) -> Path:
    root = Path(cfg.get("output_dir", "./outputs"))
    run_name = cfg.get("run_name", time.strftime("run_%Y%m%d_%H%M%S"))
    out_dir = root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, cfg, epoch, out_dir: Path):
    model.train()
    meters = {name: AverageMeter() for name in LOSS_KEYS}
    pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)
        gt, M, I_m = batch["gt"], batch["mask"], batch["masked"]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg["optim"].get("amp", False)):
            outputs = model(I_m, M)
            loss_dict = criterion(outputs, gt, M)
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["optim"].get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = gt.shape[0]
        for name, meter in meters.items():
            meter.update(float(loss_dict[name].detach()), bs)
        pbar.set_postfix({k: f"{v.avg:.4f}" for k, v in meters.items() if k in ["total", "rec", "edge"]})

    row = {"epoch": epoch, "split": "train", "lr": optimizer.param_groups[0]["lr"], "num_batches": len(loader)}
    row.update({k: v.avg for k, v in meters.items()})
    return row


@torch.no_grad()
def validate(model, loader, criterion, device, cfg, epoch, out_dir: Path, split: str = "val"):
    if loader is None:
        return None
    model.eval()
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
    fid_metric = (
        FrechetInceptionDistance(device)
        if cfg.get("eval", {}).get("compute_fid", False) and cfg.get("eval", {}).get("fid_during_train", False)
        else None
    )
    fid_bucket_metrics = (
        {bucket: FrechetInceptionDistance(device) for bucket, _, _ in MASK_RATIO_BUCKETS}
        if fid_metric is not None
        else {}
    )
    max_batches = cfg.get("eval", {}).get("max_batches")
    first_saved = False
    per_image_items = []
    completed_min = float("inf")
    completed_max = float("-inf")

    pbar = tqdm(loader, desc=f"{split} {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)
        gt, M, I_m = batch["gt"], batch["mask"], batch["masked"]
        outputs = model(I_m, M)
        loss_dict = criterion(outputs, gt, M)
        pred = outputs["pred"]

        bs = gt.shape[0]
        for name, meter in loss_meters.items():
            meter.update(float(loss_dict[name].detach()), bs)
        batch_items = evaluate_per_image(pred, gt, M, lpips_metric)
        metrics = summarize_metric_items(batch_items)
        per_image_items.extend(batch_items)
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
        show_keys = ["psnr", "ssim"]
        pbar.set_postfix({k: f"{metric_meters[k].avg:.3f}" for k in show_keys if metric_meters[k].count > 0})

        if not first_saved:
            save_triplet(I_m[:4].cpu(), pred[:4].cpu(), gt[:4].cpu(), out_dir / f"{split}_epoch_{epoch:04d}.png")
            first_saved = True

        if max_batches is not None and batch_idx >= int(max_batches):
            break

    row = {"epoch": epoch, "split": split, "num_batches": min(len(loader), int(max_batches)) if max_batches is not None else len(loader)}
    row.update({k: v.avg for k, v in loss_meters.items()})
    for name, meter in metric_meters.items():
        if meter.count > 0:
            row[name] = meter.avg
    if fid_metric is not None:
        try:
            row["fid"] = fid_metric.compute()
        except Exception as exc:
            row["fid"] = float("nan")
            print(f"[Sanity:{split}] WARNING: FID failed: {exc}")
    bucket_rows = summarize_bucket_metrics(per_image_items)
    _remove_hole_metric_columns(out_dir / "bucket_metrics.csv")
    for bucket_row in bucket_rows:
        if fid_bucket_metrics and int(bucket_row["num_samples"]) > 0:
            try:
                bucket_row["fid"] = fid_bucket_metrics[str(bucket_row["bucket"])].compute()
            except Exception as exc:
                bucket_row["fid"] = float("nan")
                print(f"[Sanity:{split}] WARNING: bucket {bucket_row['bucket']} FID failed: {exc}")
        bucket_row = {"epoch": epoch, "split": split, **bucket_row}
        append_csv_row(out_dir / "bucket_metrics.csv", bucket_row)
    for line in validation_sanity(per_image_items, completed_min, completed_max):
        print(f"[Sanity:{split}] {line}")
    row.pop("epoch")
    row.pop("split")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    out_dir = build_output_dir(cfg)
    save_config(cfg, out_dir / "config_resolved.yaml")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = FeatureGuidedElasticaADMMNet.from_config(cfg).to(device)
    print(f"Model params: {count_parameters(model) / 1e6:.2f}M")
    monitor = build_monitor(cfg, has_val=val_loader is not None)
    print(
        "Monitor: "
        f"{monitor['source']}.{monitor['metric']} ({monitor['mode']}), "
        f"patience={monitor['patience']}, min_delta={monitor['min_delta']}, "
        f"warmup_epochs={monitor['warmup_epochs']}"
    )

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

    optim_cfg = cfg["optim"]
    optimizer = AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 1e-4),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=optim_cfg.get("weight_decay", 1e-4),
    )
    total_steps = optim_cfg.get("epochs", 100) * max(len(train_loader), 1)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=optim_cfg.get("warmup_steps", 0),
        total_steps=total_steps,
        min_ratio=optim_cfg.get("min_lr_ratio", 0.05),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=optim_cfg.get("amp", False))

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        monitor_state = ckpt.get("monitor")
        if isinstance(monitor_state, dict):
            monitor.update(monitor_state)
        elif "best_score" in ckpt:
            monitor["best_score"] = ckpt["best_score"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    epochs = optim_cfg.get("epochs", 100)
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, cfg, epoch, out_dir)
        val_stats = validate(model, val_loader, criterion, device, cfg, epoch, out_dir, split="val") if val_loader is not None else None
        epoch_time_sec = time.perf_counter() - epoch_start

        monitor_stats = val_stats if monitor["source"] == "val" else train_stats
        score = monitor_value(monitor_stats, monitor)
        if score is None:
            available = sorted(monitor_stats.keys()) if monitor_stats is not None else []
            raise KeyError(f"Monitor metric '{monitor['metric']}' not found in {monitor['source']} stats: {available}")

        improved = monitor_improved(score, monitor["best_score"], str(monitor["mode"]), float(monitor["min_delta"]))
        best_before = monitor["best_score"]
        if improved:
            monitor["best_score"] = score
            monitor["best_epoch"] = epoch
            monitor["bad_epochs"] = 0
        elif epoch > int(monitor["warmup_epochs"]):
            monitor["bad_epochs"] = int(monitor["bad_epochs"]) + 1

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": monitor["best_score"],
            "monitor": monitor,
            "config": cfg,
        }
        save_checkpoint(state, out_dir / "last.pt")
        save_checkpoint(state, out_dir / "latest.pt")
        if improved:
            save_checkpoint(state, out_dir / "best.pt")
        if epoch % int(optim_cfg.get("save_every", 5)) == 0:
            save_checkpoint(state, out_dir / f"epoch_{epoch:04d}.pt")

        current_lr = optimizer.param_groups[0]["lr"]
        _remove_hole_metric_columns(out_dir / "log.csv")
        train_row = dict(train_stats)
        train_row.update(
            {
                "epoch": epoch,
                "split": "train",
                "epoch_time_sec": epoch_time_sec,
                "monitor_metric": monitor["metric"],
                "monitor_score": score,
                "best_score": monitor["best_score"],
                "best_epoch": monitor["best_epoch"],
                "bad_epochs": monitor["bad_epochs"],
                "checkpoint_best_saved": int(improved),
            }
        )
        append_csv_row(out_dir / "log.csv", train_row)
        if val_stats is not None:
            val_row = dict(val_stats)
            val_row.update(
                {
                    "epoch": epoch,
                    "split": "val",
                    "lr": current_lr,
                    "epoch_time_sec": epoch_time_sec,
                    "monitor_metric": monitor["metric"],
                    "monitor_score": score,
                    "best_score": monitor["best_score"],
                    "best_epoch": monitor["best_epoch"],
                    "bad_epochs": monitor["bad_epochs"],
                    "checkpoint_best_saved": int(improved),
                }
            )
            append_csv_row(out_dir / "log.csv", val_row)
        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] lr={current_lr:.6e} | "
            f"train: {_format_stats(train_stats, ['total', 'rec', 'edge', 'stage'])}"
        )
        if val_stats is not None:
            print(f"[Epoch {epoch:03d}/{epochs:03d}] val:   {_format_stats(val_stats, ['total', 'psnr', 'ssim', 'lpips', 'fid'])}")

        if improved:
            prev = "None" if best_before is None else f"{best_before:.4f}"
            print(
                f"[Monitor] {monitor['source']}.{monitor['metric']} improved from {prev} to {score:.4f}. "
                f"Saved best checkpoint to {(out_dir / 'best.pt').resolve()}"
            )
        else:
            if epoch <= int(monitor["warmup_epochs"]):
                print(
                    f"[Monitor] warmup epoch {epoch}/{monitor['warmup_epochs']} | "
                    f"{monitor['source']}.{monitor['metric']}={score:.4f} | patience not counting yet"
                )
            else:
                print(
                    f"[Monitor] no improvement in {int(monitor['bad_epochs'])}/{int(monitor['patience'])} epoch(s) | "
                    f"best={float(monitor['best_score']):.4f} at epoch {int(monitor['best_epoch'])}"
                )

        print(f"[Checkpoint] updated latest={(out_dir / 'latest.pt').resolve()} and last={(out_dir / 'last.pt').resolve()}")

        if int(monitor["patience"]) > 0 and int(monitor["bad_epochs"]) >= int(monitor["patience"]):
            print(
                f"[Early Stop] stopped at epoch {epoch} because "
                f"{monitor['source']}.{monitor['metric']} did not improve by more than {monitor['min_delta']} "
                f"for {monitor['patience']} consecutive epoch(s). Best epoch: {monitor['best_epoch']}."
            )
            break


if __name__ == "__main__":
    main()
