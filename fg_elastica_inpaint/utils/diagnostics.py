"""Small, explicit diagnostics for the Structure Prior V1 experiments."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..models.operators import grad, vector_norm
from .image import tensor_to_pil
from .metrics import evaluate_per_image, evaluate_structure_per_image
from .structure import edge_strength


def coupling_scale(cfg: dict, completed_epochs: float) -> float:
    schedule = cfg.get("structure_training", {})
    warmup = float(schedule.get("warmup_epochs", 0))
    ramp = float(schedule.get("ramp_epochs", 0))
    if warmup < 0 or ramp < 0:
        raise ValueError("Structure warmup/ramp epochs must be nonnegative")
    if completed_epochs < warmup:
        return 0.0
    return min(1.0, (completed_epochs - warmup) / ramp) if ramp else 1.0


def checkpoint_coupling_scale(cfg: dict, checkpoint: dict) -> float:
    if "structure_rho_scale" in checkpoint:
        return float(checkpoint["structure_rho_scale"])
    return coupling_scale(cfg, float(checkpoint["epoch"])) if "epoch" in checkpoint else 1.0


def assert_finite_outputs(outputs):
    tensors = {"pred": outputs["pred"], **outputs.get("aux", {})}
    if outputs.get("structure"):
        tensors.update({f"structure_{k}": v for k, v in outputs["structure"].items()})
    for name, value in tensors.items():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite model output: {name}")


def output_diagnostics(outputs: dict) -> dict[str, float]:
    result = {}
    for k, values in enumerate(outputs.get("diagnostics", []), 1):
        result.update({f"stage{k}_{key}": float(value) for key, value in values.items()})
    keys = ["relative_residual", "absolute_residual"]
    if bool(outputs.get("readout", {}).get("backward_solved", False)):
        keys.append("backward_relative_residual")
    for key in keys:
        value = outputs.get("readout", {}).get(key)
        if value is not None:
            result[f"readout_{key}"] = float(value.mean())
            result[f"readout_{key}_max"] = float(value.max())
    return result


def evaluate_outputs(outputs, gt, M, cfg, lpips_metric=None):
    assert_finite_outputs(outputs)
    options = {key: cfg.get("loss", {}).get(key, default) for key, default in
               (("edge_scale", .1), ("orientation_threshold", .05))}
    items = evaluate_per_image(outputs["pred"], gt, M, lpips_metric, **options)
    priors = evaluate_structure_per_image(outputs.get("structure"), gt, M, **options)
    for item, prior in zip(items, priors):
        item.update(prior)
    return items


def identified_rows(items, batch, offset=0):
    for i, item in enumerate(items):
        mask = batch["mask"][i].detach().cpu().numpy().astype(np.uint8)
        yield {**item, "index": offset+i, "path": batch["path"][i],
               "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _gray(t, path):
    a = t.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    Image.fromarray((a*255).round().astype(np.uint8)).save(path)


@torch.no_grad()
def save_structure_samples(outputs, batch, directory: Path, *, offset=0, limit=16,
                           edge_scale=.1, save_raw=False):
    """One folder per fixed sample; all gradient/edge scales are fixed across epochs."""
    count = min(batch["gt"].shape[0], max(limit-offset, 0))
    for i in range(count):
        folder = directory / f"sample_{offset+i:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        gt, M = batch["gt"][i:i+1], batch["mask"][i:i+1]
        target = grad(gt)
        pred_gradient = grad(outputs["comp"][i:i+1])
        rgb = torch.cat([batch["masked"][i:i+1], outputs["comp"][i:i+1], gt], -1)
        tensor_to_pil(rgb).save(folder / "masked_completed_gt.png")
        _gray(M, folder / "known_mask.png")
        _gray(edge_strength(target, edge_scale), folder / "edge_gt.png")
        _gray(edge_strength(pred_gradient, edge_scale), folder / "edge_completed.png")
        structure = outputs.get("structure")
        if structure:
            g = structure["gradient"][i:i+1].float()
            _gray(structure["edge"][i:i+1], folder / "edge_prior.png")
            _gray(edge_strength(g, edge_scale), folder / "edge_from_gradient_prior.png")
            # dx RGB and dy RGB, mapped from [-2,2] to RGB display range.
            signed = torch.cat([g[:, :3], g[:, 3:], target[:, :3], target[:, 3:]], -1)/2
            tensor_to_pil(signed).save(folder / "signed_Gx_Gy_GT_x_GT_y.png")
            # Signed angle encoded as hue; saturation is fixed-scale magnitude.
            for name, field in (("prior", g), ("gt", target)):
                gx, gy = field[:, :3].mean(1)[0], field[:, 3:].mean(1)[0]
                hue = (torch.atan2(gy, gx) / (2*torch.pi) + .5)
                strength = 1-torch.exp(-vector_norm(field).mean(1)[0]/edge_scale)
                hsv = torch.stack([hue, strength, torch.ones_like(hue)], -1).cpu().numpy()
                Image.fromarray((hsv*255).round().astype(np.uint8), mode="HSV").convert("RGB").save(folder/f"orientation_{name}.png")
        raw = {"gt": gt.cpu(), "mask": M.cpu(), "masked": batch["masked"][i:i+1].cpu(),
               "pred": outputs["pred"][i:i+1].detach().cpu(), "gradient_gt": target.cpu()}
        if save_raw:
            if structure:
                raw.update({f"structure_{key}": value[i:i+1].detach().cpu() for key, value in structure.items() if torch.is_tensor(value)})
            for k, state in enumerate(outputs.get("stage_states", []), 1):
                raw.update({f"stage{k}_{key}": value[i:i+1].cpu() for key, value in state.items()})
            torch.save(raw, folder / "raw_tensors.pt")
        metadata = {"path": batch["path"][i], "hole_ratio": float((1-M).mean()),
                    "edge_scale": edge_scale, "gradient_display_range": [-2, 2],
                    "diagnostics_batch_mean": output_diagnostics(outputs)}
        (folder/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
