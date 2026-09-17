"""Isolated forward-only worker; imports exactly one historical model package."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import inspect
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from compare_v0_v1 import read_json, write_json, read_input, save_prediction, valid_prediction, code_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("old", "v1"), required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = read_json(out / "protocol.json")
    entry = protocol["models"][args.model]
    project = Path(entry["project"]).resolve()
    if code_hashes(project) != entry["code"]:
        raise ValueError(f"{args.model} source changed since the comparison protocol was recorded")
    # Keep the tool directory for helpers; place the chosen project's package first.
    sys.path.insert(0, str(project))
    from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
    source = Path(inspect.getfile(FeatureGuidedElasticaADMMNet)).resolve()
    if not source.is_relative_to(project):
        raise RuntimeError(f"Wrong model package imported: {source}; expected {project}")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = entry["config"]
    # User-owned training checkpoints contain optimizer/config metadata; never
    # load an untrusted downloaded checkpoint with weights_only=False.
    checkpoint = torch.load(out / "protocol" / args.model / "checkpoint.pt", map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config")
    if saved_config is not None:
        for section in ("model", "stage_hyper"):
            if saved_config.get(section, {}) != cfg.get(section, {}):
                raise ValueError(f"{args.model}: supplied {section} differs from the checkpoint's training config")
        if saved_config.get("data", {}).get("image_size", 256) != protocol["image_size"]:
            raise ValueError("Checkpoint image_size differs from the evaluation input")
    model = FeatureGuidedElasticaADMMNet.from_config(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    epoch = checkpoint.get("epoch")
    schedule = cfg.get("structure_training", {})
    warmup, ramp = float(schedule.get("warmup_epochs", 0)), float(schedule.get("ramp_epochs", 0))
    if "structure_rho_scale" in checkpoint:
        rho_scale = float(checkpoint["structure_rho_scale"])
    elif epoch is not None:
        rho_scale = 0. if epoch < warmup else min(1., (epoch-warmup)/ramp) if ramp else 1.
    else:
        rho_scale = 1.
    if not np.isfinite(rho_scale) or rho_scale < 0:
        raise ValueError("Invalid checkpoint structure coupling scale")
    info = {"model": args.model, "source": str(source), "checkpoint_source": entry["checkpoint_source"],
            "checkpoint_sha256": entry["checkpoint_sha256"], "epoch": epoch,
            "checkpoint_config_verified": saved_config is not None,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "model_config": cfg.get("model", {}), "training_budget": cfg.get("optim", {}),
            "selection": cfg.get("monitor", {}), "stage_hyper": asdict(model.stage_hyper)
            if is_dataclass(getattr(model, "stage_hyper", None)) else cfg.get("stage_hyper", {}),
            "rho_scale": rho_scale if args.model == "v1" else None,
            "effective_rho": cfg.get("model", {}).get("structure_rho", 0)*rho_scale if args.model == "v1" else None}
    write_json(out / "protocol" / f"{args.model}_model.json", info)
    print(f"[{args.model}] loaded {source}; epoch={epoch}; params={info['parameter_count']:,}; strict=True", flush=True)
    if saved_config is None:
        print("Checkpoint has no config: supplied config used; strict weight loading alone cannot verify all numerical settings.", flush=True)
    del checkpoint
    if args.inspect:
        return
    device = torch.device(protocol["device"])
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    manifest = read_json(out / "inputs/manifest.json")
    if args.count is not None:
        manifest = manifest[:args.count]
    pending = [r for r in manifest if not valid_prediction(out / "predictions" / args.model / f"{r['index']:06d}.npz",
                                                         r["input_id"], protocol["image_size"])]
    print(f"[{args.model}] {len(pending)} pending, {len(manifest)-len(pending)} cached; FP32; device={device}", flush=True)
    started = time.perf_counter()
    batch_size = protocol["batch_size"]
    with torch.inference_mode(), tqdm(total=len(pending), desc=f"{args.model} inference", dynamic_ncols=True,
                                      ascii=True, mininterval=1) as bar:
        for start in range(0, len(pending), batch_size):
            rows = pending[start:start+batch_size]
            inputs = [read_input(out, row) for row in rows]
            gt = torch.stack([x[0] for x in inputs]).to(device)
            mask = torch.stack([x[1] for x in inputs]).to(device)
            with torch.autocast(device_type=device.type, enabled=False):
                outputs = model(gt*mask, mask, **({"rho_scale": rho_scale} if args.model == "v1" else {}))
            pred = outputs["pred"].detach().float().cpu().numpy()
            if pred.shape != tuple(gt.shape) or not np.isfinite(pred).all():
                raise FloatingPointError(f"{args.model}: invalid RGB output at sample {rows[0]['index']}")
            structure = outputs.get("structure")
            for i, row in enumerate(rows):
                extra = {}
                if structure and row["index"] < protocol["visual_count"]:
                    for key in ("gradient", "edge"):
                        array = structure[key][i].detach().float().cpu().numpy()
                        if not np.isfinite(array).all():
                            raise FloatingPointError(f"Nonfinite {key} for sample {row['index']}")
                        extra[key] = array
                save_prediction(out / "predictions" / args.model / f"{row['index']:06d}.npz",
                                pred[i], row["input_id"], **extra)
            bar.update(len(rows))
    elapsed = time.perf_counter()-started
    timing = {"scope": "preflight" if args.count is not None else "full", "new_samples": len(pending),
              "cached_samples": len(manifest)-len(pending), "seconds": elapsed,
              "seconds_per_new_sample": elapsed / len(pending) if pending else None,
              "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else None,
              "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
              "note": "Includes input loading and output writing; resumed runs have a different pending subset."}
    write_json(out / "protocol" / f"{args.model}_{timing['scope']}_timing.json", timing)
    print(f"[{args.model}] completed {len(pending)} new images in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
