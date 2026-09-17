from __future__ import annotations

import argparse
import json
import subprocess
import time
from itertools import islice
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW

from fg_elastica_inpaint.data.dataset import build_dataloaders
from fg_elastica_inpaint.losses import InpaintingLoss
from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.diagnostics import (coupling_scale, output_diagnostics, evaluate_outputs,
    identified_rows, write_rows, save_structure_samples, assert_finite_outputs)
from fg_elastica_inpaint.utils.config import load_config, save_config
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
from fg_elastica_inpaint.utils.misc import AverageMeter, MaximumMeter, append_csv_row, count_parameters, save_checkpoint, set_seed
from fg_elastica_inpaint.utils.scheduler import build_warmup_cosine_scheduler
from fg_elastica_inpaint.utils.progress import progress_bar


def _format_stats(stats: Optional[Dict], keys) -> str:
    if not stats:
        return "n/a"
    parts = []
    for key in keys:
        if key in stats:
            parts.append(f"{key}={stats[key]:.4f}")
    return ", ".join(parts) if parts else "n/a"


def build_monitor(cfg: Dict, has_val: bool) -> Dict[str, object]:
    monitor_cfg = cfg.get("monitor", {})
    metric = monitor_cfg.get("metric")
    if metric is None:
        metric = "psnr" if has_val else "total"

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
    meters = {}
    optimizer_steps = amp_skipped_steps = consecutive_overflows = 0
    if not len(loader):
        raise ValueError("Training loader is empty: check train_list, limit and batch_size")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pbar = progress_bar(loader, desc=f"Epoch {epoch}/{cfg['optim'].get('epochs', '?')}", cfg=cfg)
    for step, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)
        gt, M, I_m = batch["gt"], batch["mask"], batch["masked"]
        optimizer.zero_grad(set_to_none=True)
        rho_scale = coupling_scale(cfg, epoch - 1 + (step - 1) / len(loader))
        with torch.amp.autocast(device.type, enabled=device.type == "cuda" and cfg["optim"].get("amp", False)):
            outputs = model(I_m, M, rho_scale=rho_scale)
            assert_finite_outputs(outputs)
            loss_dict = criterion(outputs, gt, M)
            loss = loss_dict["total"]

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at epoch={epoch}, step={step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        prior_grad = sum(float(p.grad.detach().square().sum()) for p in model.structure_prior.parameters()
                         if p.grad is not None) ** .5 if model.structure_prior is not None else 0.0
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg["optim"].get("grad_clip", 1.0), error_if_nonfinite=True)
        except RuntimeError as error:
            message = str(error)
            if not message.startswith("The total norm of order ") or "is non-finite" not in message:
                raise  # Preserve unrelated clipping failures, including out-of-memory errors.
            # Clipping raises before modifying gradients. unscale_ has already
            # recorded AMP overflow, so GradScaler can safely skip this update.
            named_grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
            finite = torch.stack([torch.isfinite(g).all() for _, g in named_grads]).tolist() if named_grads else []
            bad_names = [name for (name, _), ok in zip(named_grads, finite) if not ok]
            del named_grads  # Do not retain discarded gradient buffers for the rest of the epoch.
            if not bad_names:
                raise  # A different clipping failure, including finite-gradient norm overflow.
            detail = f"epoch={epoch}, step={step}, parameters={', '.join(bad_names[:8])}"
            if not scaler.is_enabled():
                raise FloatingPointError(f"Nonfinite gradients without AMP scaling: {detail}") from None
            scale_before = scaler.get_scale()
            scaler.step(optimizer)  # The recorded nonfinite gradients prevent optimizer.step().
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            amp_skipped_steps += 1
            consecutive_overflows += 1
            append_csv_row(out_dir / "amp_events.csv", {
                "epoch": epoch, "step": step, "scale_before": scale_before,
                "scale_after": scaler.get_scale(), "bad_parameters": ";".join(bad_names),
                "paths": json.dumps(batch.get("path", []), ensure_ascii=False),
            })
            pbar.write(f"[AMP] Skipped update ({detail}); scale {scale_before:g} -> {scaler.get_scale():g}")
            if consecutive_overflows >= 8:
                raise FloatingPointError(f"Persistent nonfinite gradients after 8 consecutive AMP backoffs: {detail}") from None
            continue  # Do not advance the scheduler or include invalid gradient diagnostics.
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer_steps += 1
        consecutive_overflows = 0

        bs = gt.shape[0]
        values = {name: float(value.detach()) for name, value in loss_dict.items()}
        values.update(output_diagnostics(outputs))
        values.update(rho_scale=rho_scale, structure_grad_norm=prior_grad, grad_norm=float(grad_norm))
        for name, value in values.items():
            meters.setdefault(name, MaximumMeter() if name.endswith("_max") else AverageMeter()).update(value, bs)
        pbar.set_postfix(loss=f"{meters['total'].avg:.4f}", refresh=False)

    if not optimizer_steps:
        raise FloatingPointError(f"No optimizer updates in epoch={epoch}; inspect amp_events.csv before continuing")
    row = {"epoch": epoch, "split": "train", "lr": optimizer.param_groups[0]["lr"], "num_batches": len(loader),
           "optimizer_steps": optimizer_steps, "amp_skipped_steps": amp_skipped_steps, "amp_scale": scaler.get_scale()}
    row.update({k: v.avg for k, v in meters.items()})
    if device.type == "cuda":
        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        row["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    return row


@torch.no_grad()
def validate(model, loader, criterion, device, cfg, epoch, out_dir: Path, split: str = "val"):
    if loader is None:
        return None
    model.eval()
    loss_meters = {}
    metric_meters = {}
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
    diagnostic_cfg = cfg.get("diagnostics", {})
    sample_limit = int(diagnostic_cfg.get("fixed_samples", 16))
    save_raw = epoch == 1 or epoch % int(diagnostic_cfg.get("raw_every", 5)) == 0
    saved = 0
    named_items = []
    per_image_items = []
    completed_min = float("inf")
    completed_max = float("-inf")

    batch_count = min(len(loader), max(1, int(max_batches))) if max_batches is not None else len(loader)
    pbar = progress_bar(islice(loader, batch_count), total=batch_count,
                        desc=f"{split.capitalize()} {epoch}", cfg=cfg)
    for batch_idx, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)
        gt, M, I_m = batch["gt"], batch["mask"], batch["masked"]
        outputs = model(I_m, M, rho_scale=coupling_scale(cfg, epoch),
                        return_stage_states=save_raw and saved < sample_limit)
        loss_dict = criterion(outputs, gt, M)
        pred = outputs["pred"]

        bs = gt.shape[0]
        for name, value in loss_dict.items():
            loss_meters.setdefault(name, AverageMeter()).update(float(value.detach()), bs)
        batch_items = evaluate_outputs(outputs, gt, M, cfg, lpips_metric)
        named_items.extend(identified_rows(batch_items, batch, len(per_image_items)))
        metrics = summarize_metric_items(batch_items)
        per_image_items.extend(batch_items)
        for name, value in metrics.items():
            metric_meters.setdefault(name, MaximumMeter() if name.endswith("_max") else AverageMeter()).update(value, bs)
        for name, value in output_diagnostics(outputs).items():
            metric_meters.setdefault(name, MaximumMeter() if name.endswith("_max") else AverageMeter()).update(value, bs)
        completed = composite_completed(pred, gt, M)
        completed_min = min(completed_min, float(completed.min().detach()))
        completed_max = max(completed_max, float(completed.max().detach()))
        if fid_metric is not None:
            fid_metric.update(completed, gt)
            for sample_idx, item in enumerate(batch_items):
                bucket = mask_ratio_bucket(float(item["hole_ratio"]))
                fid_bucket_metrics[bucket].update(completed[sample_idx : sample_idx + 1], gt[sample_idx : sample_idx + 1])
        show_keys = ["psnr", "ssim"]
        pbar.set_postfix({k: f"{metric_meters[k].avg:.3f}" for k in show_keys if metric_meters[k].count > 0}, refresh=False)

        if saved < sample_limit:
            save_structure_samples(outputs, batch, out_dir / "diagnostics" / f"{split}_{epoch:04d}",
                                   offset=saved, limit=sample_limit, save_raw=save_raw,
                                   edge_scale=cfg.get("loss", {}).get("edge_scale", .1))
            saved += bs

    row = {"epoch": epoch, "split": split, "num_batches": min(len(loader), int(max_batches)) if max_batches is not None else len(loader)}
    row.update({k: v.avg for k, v in loss_meters.items()})
    for name, meter in metric_meters.items():
        if meter.count > 0:
            row[name] = meter.avg
    # Aggregate supported per-image regions, not means of unequal batch supports.
    row.update(summarize_metric_items(per_image_items))
    if fid_metric is not None:
        try:
            row["fid"] = fid_metric.compute()
        except Exception as exc:
            row["fid"] = float("nan")
            print(f"[Sanity:{split}] WARNING: FID failed: {exc}")
    bucket_rows = summarize_bucket_metrics(per_image_items)
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
    write_rows(out_dir / "per_image" / f"{split}_{epoch:04d}.csv", named_items)
    row.pop("epoch")
    row.pop("split")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-scheduler", action="store_true",
                        help="Keep resumed weights and Adam state, but start a new configured LR schedule")
    args = parser.parse_args()
    if args.reset_scheduler and not args.resume:
        parser.error("--reset-scheduler requires --resume")

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
    # Optional head initialization must not change data-order RNG in R0/R1/R2.
    set_seed(int(cfg.get("seed", 42)))
    print(f"Model params: {count_parameters(model) / 1e6:.2f}M")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    metadata = {"git_revision": revision, "torch": torch.__version__, "device": str(device),
                "parameters": count_parameters(model), "config_path": str(Path(args.config).resolve()),
                "prior_parameters": count_parameters(model.structure_prior) if model.structure_prior else 0}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(gpu=properties.name, dedicated_memory_gib=properties.total_memory/2**30)
    (out_dir/"run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    monitor = build_monitor(cfg, has_val=val_loader is not None)
    print(
        "Monitor: "
        f"{monitor['source']}.{monitor['metric']} ({monitor['mode']}), "
        f"patience={monitor['patience']}, min_delta={monitor['min_delta']}, "
        f"warmup_epochs={monitor['warmup_epochs']}"
    )

    criterion = InpaintingLoss.from_config(cfg).to(device)

    optim_cfg = cfg["optim"]
    optimizer = AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 1e-4),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=optim_cfg.get("weight_decay", 1e-4),
    )
    schedule_start_epoch = int(optim_cfg.get("schedule_start_epoch", 0))
    schedule_epochs = int(optim_cfg.get("epochs", 100)) - schedule_start_epoch
    if schedule_start_epoch < 0 or schedule_epochs <= 0:
        raise ValueError("Schedule must span at least one epoch after optim.schedule_start_epoch")
    total_steps = schedule_epochs * max(len(train_loader), 1)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=optim_cfg.get("warmup_steps", 0),
        total_steps=total_steps,
        min_ratio=optim_cfg.get("min_lr_ratio", 0.05),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and optim_cfg.get("amp", False))

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if args.reset_scheduler:
            if schedule_start_epoch != int(ckpt["epoch"]):
                raise ValueError("Reset schedule requires optim.schedule_start_epoch == checkpoint epoch")
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] = float(optim_cfg["lr"])
            scheduler = build_warmup_cosine_scheduler(
                optimizer, warmup_steps=optim_cfg.get("warmup_steps", 0),
                total_steps=total_steps, min_ratio=optim_cfg.get("min_lr_ratio", 0.05))
            print(f"[Continuation] Preserved Adam state; reset LR schedule for {total_steps} updates, "
                  f"starting lr={optimizer.param_groups[0]['lr']:.6e}")
        else:
            saved_start = int(ckpt.get("config", {}).get("optim", {}).get("schedule_start_epoch", 0))
            if schedule_start_epoch != saved_start:
                raise ValueError("Schedule origin changed; use --reset-scheduler explicitly")
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
        # Reproducible epoch-level shuffling, augmentation and mask streams, also on resume.
        set_seed(int(cfg.get("seed", 42)) + epoch)
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

        structure_score = val_stats.get("hole_edge_f1") if val_stats else None
        structure_improved = structure_score is not None and structure_score > monitor.get("best_structure_score", -float("inf"))
        if structure_improved:
            monitor["best_structure_score"] = structure_score
            monitor["best_structure_epoch"] = epoch
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": monitor["best_score"],
            "monitor": monitor,
            "config": cfg,
            "structure_rho_scale": coupling_scale(cfg, epoch),
        }
        save_checkpoint(state, out_dir / "last.pt")
        save_checkpoint(state, out_dir / "latest.pt")
        if improved:
            save_checkpoint(state, out_dir / "best.pt")
        if structure_improved:
            save_checkpoint(state, out_dir / "best_structure.pt")
        if epoch % int(optim_cfg.get("save_every", 5)) == 0:
            save_checkpoint(state, out_dir / f"epoch_{epoch:04d}.pt")

        current_lr = optimizer.param_groups[0]["lr"]
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
