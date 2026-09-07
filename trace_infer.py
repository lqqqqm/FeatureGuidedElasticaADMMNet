"""Export tensors from the actual model forward, including final PCG readout."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
import torch

from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.models.operators import grad, vector_norm
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.image import pil_to_tensor, tensor_to_pil
from fg_elastica_inpaint.utils.structure import edge_strength


def load_rgb(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return pil_to_tensor(image, value_range="minus_one_to_one").unsqueeze(0)


def load_mask(path: str, size: int, invert: bool = False) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        array = (np.asarray(image, dtype=np.float32) > 127.5).astype(np.float32)
    if invert:
        array = 1.0 - array
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def save_rgb(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(t).save(path)


def _normalize01(x: torch.Tensor, value_range: tuple[float, float] | None = None) -> torch.Tensor:
    x = x.detach().float().cpu()
    if value_range is None:
        lo, hi = x.amin(), x.amax()
        return (x - lo) / (hi - lo).clamp_min(1e-8)
    lo, hi = value_range
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError("Display range must contain two increasing finite bounds.")
    return ((x - lo) / (hi - lo)).clamp(0, 1)


def save_gray_map(x: torch.Tensor, path: Path, value_range: tuple[float, float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.mean(dim=0)
    array = (_normalize01(x, value_range).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def save_signed_map(x: torch.Tensor, path: Path, vmax: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.mean(dim=0)
    scale = x.abs().amax().clamp_min(1e-8) if vmax is None else vmax
    red, blue = (x.clamp_min(0) / scale).clamp_max(1), ((-x).clamp_min(0) / scale).clamp_max(1)
    green = 1 - (red + blue).clamp(0, 1)
    array = (torch.stack([red, green, blue], dim=-1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def save_channel_grid(x: torch.Tensor, path: Path, max_channels: int = 16,
                      value_range: tuple[float, float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    channels = min(int(x.shape[0]), max_channels)
    maps = [_normalize01(x[i], value_range) for i in range(channels)]
    h, w = maps[0].shape
    cols = 3 if channels == 6 else 4
    grid = torch.zeros(((channels + cols - 1) // cols) * h, cols * w)
    for idx, item in enumerate(maps):
        row, col = divmod(idx, cols)
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = item
    Image.fromarray((grid.numpy() * 255).round().astype(np.uint8)).save(path)


def vector_magnitude(x: torch.Tensor) -> torch.Tensor:
    return vector_norm(x) if x.shape[1] % 2 == 0 else x.abs()


def tensor_stats(t: Optional[torch.Tensor]) -> Optional[Dict[str, object]]:
    if t is None:
        return None
    x = t.detach().float().cpu()
    return {"shape": list(x.shape), "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()), "std": float(x.std(unbiased=False)),
            "l2": float(torch.linalg.vector_norm(x))}


def add_tensor(trace: Dict[str, torch.Tensor], stats: Dict[str, object],
               name: str, value: Optional[torch.Tensor]) -> None:
    if value is None:
        stats[name] = None
        return
    trace[name] = value.detach().cpu()
    stats[name] = tensor_stats(value)


def _collect_nested(trace, stats, prefix, value) -> None:
    if torch.is_tensor(value) or value is None:
        add_tensor(trace, stats, prefix, value)
    elif isinstance(value, dict):
        for name, item in value.items():
            _collect_nested(trace, stats, f"{prefix}.{name}", item)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _collect_nested(trace, stats, f"{prefix}.{index}", item)
    else:
        stats[prefix] = value


def collect_trace(outputs: dict, gt: torch.Tensor, M: torch.Tensor, *, edge_scale: float = .1,
                  prior_gradient: torch.Tensor | None = None):
    """Flatten returned tensors; GT-derived targets are diagnostic-only data."""
    trace, stats = {}, {}
    for name, value in (("input", gt), ("mask", M), ("masked", gt * M),
                        ("u_0", gt * M), ("pred", outputs["pred"]), ("comp", outputs["comp"])):
        add_tensor(trace, stats, name, value)
    for index, state in enumerate(outputs["stage_states"], start=1):
        _collect_nested(trace, stats, f"stage_{index}", state)
    for index, diagnostic in enumerate(outputs["diagnostics"], start=1):
        _collect_nested(trace, stats, f"diagnostics.stage_{index}", diagnostic)
    for index, prediction in enumerate(outputs["stage_preds"], start=1):
        add_tensor(trace, stats, f"stage_pred_{index}", prediction)
    for key in ("structure", "readout", "aux"):
        _collect_nested(trace, stats, key, outputs.get(key))
    target_gradient = grad(gt.float())
    add_tensor(trace, stats, "target.gradient", target_gradient)
    add_tensor(trace, stats, "target.edge", edge_strength(target_gradient, scale=edge_scale))
    if prior_gradient is None and outputs.get("structure") is not None:
        prior_gradient = outputs["structure"]["gradient"]
    if prior_gradient is not None:
        add_tensor(trace, stats, "prior.gradient", prior_gradient)
        add_tensor(trace, stats, "prior.edge", edge_strength(prior_gradient, scale=edge_scale))
    stats["_metadata"] = {
        "source": "Actual model forward with return_stage_states=True",
        "mask_convention": "1=known, 0=hole",
        "target_use": "GT gradients/edges are saved only for diagnostics.",
        "edge_scale": edge_scale,
        "display_ranges": {"gradient_components": [-2., 2.], "edge": [0., 1.],
                           "gradient_magnitude": [0., math.sqrt(8.)]},
        "feature_maps": "Only tensors exposed by the actual forward are exported.",
    }
    return trace, stats


@torch.no_grad()
def trace_forward(model: FeatureGuidedElasticaADMMNet, gt: torch.Tensor, M: torch.Tensor,
                  *, edge_scale: float = .1, **forward_options):
    outputs = model(gt * M, M, return_stage_states=True, **forward_options)
    return collect_trace(outputs, gt, M, edge_scale=edge_scale,
                         prior_gradient=forward_options.get("prior_gradient"))


def write_visualizations(trace: Dict[str, torch.Tensor], out_dir: Path) -> None:
    images = out_dir / "images"
    for name in ("input", "mask", "masked", "u_0", "pred", "comp"):
        if name in trace:
            if name == "mask":
                save_gray_map(trace[name], images / f"{name}.png", value_range=(0., 1.))
            else:
                save_rgb(trace[name], images / f"{name}.png")
    if "readout.u_before" in trace:
        save_rgb(trace["readout.u_before"], images / "before_readout.png")
    index = 1
    while f"stage_{index}.u" in trace:
        for suffix in ("u_tilde", "u"):
            key = f"stage_{index}.{suffix}"
            if key in trace:
                save_rgb(trace[key], images / f"stage_{index}_{suffix}.png")
        key = f"stage_{index}.correction_u"
        if key in trace:
            save_signed_map(trace[key], images / f"stage_{index}_du_signed.png")
        for suffix in ("p_tilde", "p", "m", "n_tilde", "n", "p_baseline"):
            key = f"stage_{index}.{suffix}"
            if key in trace:
                save_gray_map(vector_magnitude(trace[key]), out_dir / "vectors" / f"stage_{index}_{suffix}_magnitude.png")
        for suffix in ("lambda1", "lambda2", "lambda4"):
            key = f"stage_{index}.{suffix}"
            if key in trace:
                save_signed_map(trace[key], out_dir / "lambdas" / f"stage_{index}_{suffix}_signed.png")
        index += 1
    # Six gradient components retain a common [-2,2] range across all modes.
    for prefix in ("structure", "target", "prior"):
        if f"{prefix}.gradient" in trace:
            gradient = trace[f"{prefix}.gradient"]
            directory = out_dir / prefix
            save_channel_grid(gradient, directory / "gradient_components.png", 6, value_range=(-2., 2.))
            save_rgb(gradient[:, :3] / 2, directory / "gradient_dx_rgb.png")
            save_rgb(gradient[:, 3:] / 2, directory / "gradient_dy_rgb.png")
            save_gray_map(vector_norm(gradient), directory / "gradient_magnitude.png",
                          value_range=(0., math.sqrt(8.)))
        if f"{prefix}.edge" in trace:
            save_gray_map(trace[f"{prefix}.edge"], out_dir / prefix / "edge.png", value_range=(0., 1.))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "image", "mask", "output_dir"):
        parser.add_argument(f"--{name}", type=str, required=True)
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    size = int(cfg["data"].get("image_size", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    model = FeatureGuidedElasticaADMMNet.from_config(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    gt = load_rgb(args.image, size).to(device)
    M = load_mask(args.mask, size, invert=args.invert_mask).to(device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_scale = float(cfg.get("loss", {}).get("edge_scale", .1))
    from fg_elastica_inpaint.utils.diagnostics import checkpoint_coupling_scale
    trace, stats = trace_forward(model, gt, M, edge_scale=edge_scale,
                                 rho_scale=checkpoint_coupling_scale(cfg, checkpoint))
    stats["_metadata"]["sources"] = {key: str(Path(getattr(args, key)).resolve())
                                      for key in ("config", "checkpoint", "image", "mask")}
    write_visualizations(trace, out_dir)
    torch.save(trace, out_dir / "raw_tensors.pt")
    with (out_dir / "stats.json").open("w", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved trace outputs to {out_dir.resolve()}")
    print(f"Saved {len(trace)} tensors and stats for {len(stats)} entries.")


if __name__ == "__main__":
    main()
