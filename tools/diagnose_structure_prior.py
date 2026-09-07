"""Run five labelled inference interventions on one fixed image and mask.

Oracle GT gradients are diagnostic inputs, never a deployable performance
result. This tool neither trains nor initializes downloadable learned metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.models.operators import grad
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.metrics import evaluate_per_image, evaluate_structure_per_image
from fg_elastica_inpaint.utils.structure import edge_strength
from trace_infer import collect_trace, load_mask, load_rgb, write_visualizations


def spatial_shuffle_gradient(gradient: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Permute full-image spatial vectors, preserving all six channels together.

    This is a single-image spatial shuffle, not a shuffle between batch items.
    A private CPU RNG leaves the global training/inference RNG state untouched.
    """
    if gradient.ndim != 4 or gradient.shape[0] != 1 or gradient.shape[1] != 6:
        raise ValueError("Spatial-shuffle diagnostic requires one [1,6,H,W] gradient.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(gradient.shape[-2] * gradient.shape[-1], generator=generator)
    return gradient.flatten(2).index_select(2, permutation.to(gradient.device)).reshape_as(gradient)


def _json_tensors(value):
    if torch.is_tensor(value):
        cpu = value.detach().cpu()
        return float(cpu) if cpu.numel() == 1 else cpu.tolist()
    if isinstance(value, dict):
        return {key: _json_tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_tensors(item) for item in value]
    return value


def _output_change(prediction: torch.Tensor, baseline: torch.Tensor, M: torch.Tensor) -> dict:
    hole = (1 - M).expand_as(prediction)
    diff = (prediction - baseline) * hole
    hole_l1 = diff.abs().sum() / hole.sum().clamp_min(1)
    relative_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(baseline * hole).clamp_min(1e-8)
    return {"hole_l1": float(hole_l1), "hole_relative_l2": float(relative_l2),
            "max_absolute": float(diff.abs().max())}


@torch.no_grad()
def run_diagnostics(
    model: FeatureGuidedElasticaADMMNet,
    gt: torch.Tensor,
    M: torch.Tensor,
    output_dir: str | Path,
    *,
    seed: int = 42,
    edge_scale: float = .1,
    orientation_threshold: float = .05,
    provenance: dict | None = None,
) -> dict:
    """Run actual forwards without changing model parameters or configuration.

    Each directory stores original decoder G/E under structure/, the effective
    injected G and its implied edge under prior/, GT targets under target/,
    raw tensors, image metrics, scalar stage diagnostics and metric differences
    from the predicted-prior baseline. A disabled prior still displays its
    decoded candidate field, with active=false explicitly recorded.
    """
    if model.training:
        raise ValueError("Diagnostics require model.eval() for repeatable inference.")
    if gt.ndim != 4 or gt.shape[0] != 1 or M.shape != gt[:, :1].shape:
        raise ValueError("Use exactly one RGB image and its matching [1,1,H,W] known mask.")
    if model.structure_prior is None:
        raise ValueError("The five-mode diagnostic requires a model with use_structure_prior=true.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masked = gt * M
    baseline = model(masked, M, return_stage_states=True)
    predicted_gradient = baseline["structure"]["gradient"].detach()
    oracle_gradient = grad(gt.float())
    shuffled_gradient = spatial_shuffle_gradient(predicted_gradient, seed=seed)
    modes = [
        ("predicted", {}, predicted_gradient, "Predicted decoder G, configured rho."),
        ("disabled_prior", {"rho_scale": 0.}, predicted_gradient, "Disable only prior injection with rho_scale=0."),
        ("spatial_shuffled", {"prior_gradient": shuffled_gradient}, shuffled_gradient,
         "Single-image full-spatial shuffle of predicted six-channel G vectors; not a batch shuffle."),
        ("oracle_gt_gradient", {"prior_gradient": oracle_gradient}, oracle_gradient,
         "DIAGNOSTIC ONLY: inject GT gradient. This oracle is not a deployable performance result."),
        ("disable_correction", {"disable_correction": True}, predicted_gradient,
         "Disable Correction modules during this forward while retaining predicted prior injection."),
    ]
    report = {
        "purpose": "Single fixed-image inference interventions; no training or weight/configuration changes.",
        "oracle_is_diagnostic_only": True,
        "shuffle_kind": "single_image_spatial_vectors_not_batch",
        "shuffle_seed": seed,
        "configured_rho": model.structure_rho,
        "edge_scale": edge_scale,
        "orientation_threshold": orientation_threshold,
        "lpips_note": "Not initialized by this offline diagnostic; lpips_available=0, no pretrained downloads.",
        "delta_convention": "Metric value minus predicted-baseline value; no percentage or ranking claim.",
        "relative_output_convention": "Hole L2 difference divided by baseline hole L2 norm, with denominator floor 1e-8.",
        "sources": provenance or {},
        "modes": {},
    }
    baseline_metrics = None
    for name, options, effective_gradient, description in modes:
        outputs = baseline if name == "predicted" else model(masked, M, return_stage_states=True, **options)
        trace, stats = collect_trace(outputs, gt, M, edge_scale=edge_scale, prior_gradient=effective_gradient)
        metrics = evaluate_per_image(outputs["pred"], gt, M, edge_scale=edge_scale,
                                     orientation_threshold=orientation_threshold)[0]
        metrics.update(evaluate_structure_per_image(outputs["structure"], gt, M,
                       edge_scale=edge_scale, orientation_threshold=orientation_threshold)[0])
        # Keep decoder scores distinct from the field actually used in the p update.
        effective_structure = {"gradient": effective_gradient,
                               "edge": edge_strength(effective_gradient, scale=edge_scale)}
        effective_metrics = evaluate_structure_per_image(effective_structure, gt, M,
                            edge_scale=edge_scale, orientation_threshold=orientation_threshold)[0]
        metrics.update({key.replace("structure_", "effective_prior_", 1): value
                        for key, value in effective_metrics.items()})
        if baseline_metrics is None:
            baseline_metrics = metrics
        deltas = {key: value - baseline_metrics[key] for key, value in metrics.items()
                  if key in baseline_metrics and key not in {"index", "hole_ratio"}
                  and not key.endswith(("_count", "_pixels", "_available"))}
        entry = {
            "description": description,
            "prior_active": model.structure_rho * options.get("rho_scale", 1.) > 0,
            "rho_scale": options.get("rho_scale", 1.),
            "correction_disabled": options.get("disable_correction", False),
            "oracle_is_diagnostic_only": name == "oracle_gt_gradient",
            "metrics": metrics,
            "delta_from_predicted": deltas,
            "output_change": _output_change(outputs["comp"], baseline["comp"], M),
            "diagnostics": _json_tensors(outputs["diagnostics"]),
            "readout": _json_tensors({key: value for key, value in outputs["readout"].items() if key != "u_before"}),
        }
        report["modes"][name] = entry
        stats["_intervention"] = {key: value for key, value in entry.items()
                                  if key not in {"metrics", "delta_from_predicted", "diagnostics", "readout"}}
        mode_dir = output_dir / name
        mode_dir.mkdir(parents=True, exist_ok=True)
        write_visualizations(trace, mode_dir)
        torch.save(trace, mode_dir / "raw_tensors.pt")
        with (mode_dir / "stats.json").open("w", encoding="utf-8") as stream:
            json.dump(stats, stream, indent=2, allow_nan=False)
        with (mode_dir / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(entry, stream, indent=2, allow_nan=False)
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "image", "mask", "output_dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    model = FeatureGuidedElasticaADMMNet.from_config(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    size = int(cfg["data"].get("image_size", 256))
    gt = load_rgb(args.image, size).to(device)
    M = load_mask(args.mask, size, invert=args.invert_mask).to(device)
    run_diagnostics(model, gt, M, args.output_dir, seed=args.seed,
                    edge_scale=float(cfg.get("loss", {}).get("edge_scale", .1)),
                    orientation_threshold=float(cfg.get("loss", {}).get("orientation_threshold", .05)),
                    provenance={key: str(Path(getattr(args, key)).resolve())
                                for key in ("config", "checkpoint", "image", "mask")})
    print(f"Saved five single-image diagnostic modes to {Path(args.output_dir).resolve()}")
    print("Oracle uses GT gradients and is diagnostic only, not a deployable performance result.")


if __name__ == "__main__":
    main()
