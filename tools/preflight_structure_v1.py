"""Training-only scale and PCG diagnostics; never train or modify a config.

CPU self-check, runnable from any working directory using absolute paths:
  python tools/preflight_structure_v1.py --config configs/mvp.yaml --samples 1 \
      --device cpu --synthetic --synthetic-size 32 --output preflight.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from fg_elastica_inpaint.data.dataset import InpaintingImageDataset, read_file_list
from fg_elastica_inpaint.data.mask_generator import RandomMaskGenerator
from fg_elastica_inpaint.models.network import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.models.operators import dot_mp, grad, vector_norm
from fg_elastica_inpaint.models.solvers import _pcg, poisson_operator, solve_u_pcg
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.structure import edge_strength

FORWARD_BUDGETS = (20, 40, 80, 160, 320, 512)
BACKWARD_BUDGETS = (160, 320, 640)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _resolve(path: str | Path, base: Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    choices = ([base / path] if base is not None else []) + [Path.cwd() / path, PROJECT_ROOT / path]
    return next((item.resolve() for item in choices if item.exists()), choices[0].resolve())


def load_samples(cfg: Mapping, count: int, *, seed: int, synthetic: bool = False,
                 synthetic_size: int = 32, config_path: Path | None = None) -> list[dict]:
    """Select train_list only, then freeze augmented GT and random masks by seed."""
    if count < 1:
        raise ValueError("--samples must be positive")
    data = cfg.get("data", {})
    size = synthetic_size if synthetic else data.get("image_size", 256)
    if size < 8 or size % 8:
        raise ValueError("Image size must be a positive multiple of 8")
    options = dict(hole_buckets=data.get("hole_buckets"), bucket_probs=data.get("bucket_probs"),
                   mask_topology_modes=data.get("mask_topology_modes"),
                   mask_topology_probs=data.get("mask_topology_probs"))
    if synthetic:
        generator = RandomMaskGenerator(size=size, hole_buckets=options["hole_buckets"],
            bucket_probs=options["bucket_probs"], topology_modes=options["mask_topology_modes"],
            topology_probs=options["mask_topology_probs"])
        dataset = None
    else:
        if not data.get("train_list"):
            raise ValueError("data.train_list is required; use --synthetic explicitly for a CPU self-check")
        base = config_path.parent if config_path is not None else None
        train_list = _resolve(data["train_list"], base)
        if data.get("test_list") and train_list == _resolve(data["test_list"], base):
            raise ValueError("train_list equals test_list; test-set calibration is forbidden")
        paths = read_file_list(train_list, limit=data.get("train_limit"))
        if not paths:
            raise ValueError(f"No training images in train_list: {train_list}")
        paths = random.Random(seed).sample(paths, min(count, len(paths)))
        paths = [str(_resolve(path, train_list.parent)) for path in paths]
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Training image does not exist: {path}")
        dataset = InpaintingImageDataset(paths, image_size=size, train=True,
            resize_short_to=max(size, data.get("resize_short_to", 286)), **options)
        count = len(dataset)
    samples = []
    for index in range(count):
        sample_seed = seed + index * 1009
        _seed(sample_seed)
        if dataset is None:
            y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
            phase = index * 0.37
            texture = 0.5 * torch.sin(4 * math.pi * x + phase) * torch.cos(3 * math.pi * y)
            gt = torch.stack((x, y, texture))
            gt[0, size // 4:3 * size // 4, size // 4:3 * size // 4] += 0.5
            item = {"gt": gt.clamp(-1, 1), "mask": generator(size, size), "path": f"synthetic:{index}"}
        else:
            item = dataset[index]
        gt, mask = item["gt"].unsqueeze(0).float(), item["mask"].unsqueeze(0).float()
        if not bool(torch.isfinite(gt).all()) or not 0 < mask.sum().item() < mask.numel():
            raise ValueError(f"Sample needs finite GT, known pixels, and holes: {item['path']}")
        samples.append({"gt": gt, "mask": mask, "masked": gt * mask,
            "id": str(item["path"]), "seed": sample_seed,
            "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest()})
    return samples


def prior_activation_threshold(q: torch.Tensor, c: torch.Tensor, gradient: torch.Tensor,
                               r2: float) -> torch.Tensor:
    """Critical rho for initially inactive vectors: |r2*q + rho*G| > c.

    Solve the positive quadratic root, including opposition between q and G.
    In the flat q=0 case this is c/|G|: increasing the denominator alone
    cannot overcome shrinkage. Already active vectors return zero.
    """
    a = r2 * q
    norm_a_sq = vector_norm(a).square()
    norm_g_sq = vector_norm(gradient).square()
    dot = dot_mp(a, gradient)
    discriminant = dot.square() + norm_g_sq * (c.square() - norm_a_sq)
    root = (-dot + discriminant.clamp_min(0).sqrt()) / norm_g_sq.clamp_min(1e-20)
    root = torch.where(norm_a_sq > c.square(), 0, root.clamp_min(0))
    return torch.where(norm_g_sq > 0, root, torch.full_like(root, float("inf")))


def _quantiles(values: list[torch.Tensor] | torch.Tensor) -> dict:
    values = torch.cat([x.detach().float().cpu().reshape(-1) for x in values]) if isinstance(values, list) else values.detach().float().cpu().reshape(-1)
    values = values[torch.isfinite(values)]
    if not values.numel():
        return {"count": 0, **{f"p{p}": None for p in (10, 25, 50, 75, 90, 95, 99)}}
    points = torch.tensor([.10, .25, .50, .75, .90, .95, .99])
    return {"count": values.numel(), **{f"p{round(100*p)}": float(v) for p, v in zip(points.tolist(), torch.quantile(values, points))}}


def _hole_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    hole = (1 - mask).expand_as(value)
    return float((value * hole).sum() / hole.sum().clamp_min(1))


def _residual_info(info: Mapping) -> dict:
    return {"relative_residual_mean": float(info["relative_residual"].mean()),
            "relative_residual_max": float(info["relative_residual"].max()),
            "absolute_residual_max": float(info["absolute_residual"].max())}


def backward_sweep(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                   r2: float, eta: float, reference_iterations: int, tolerance: float) -> dict:
    """Calibrate the production implicit backward solve for a hole RGB L1 loss.

    The upstream gradient is sign(pred-GT)*H/(3*sum(H)), so its small scale
    matches a normalized training loss. Compare A*v=gradient using the same
    production _pcg and operator as _PoissonSolve.backward. The reference
    casts the same FP32 RHS and mask to FP64. VJP differences use the actual
    FP32 grad(adjoint) calculation before casting for comparison.
    """
    pred, gt, mask = pred.float(), gt.float(), mask.float()
    hole = (1 - mask).expand_as(pred)
    upstream = (pred - gt).sign() * hole / hole.sum().clamp_min(1)
    zero = torch.zeros_like(upstream)
    reference_iterations = max(reference_iterations, 1024)
    reference_tolerance = min(tolerance, 1e-10)
    tiny = torch.finfo(torch.float64).tiny

    def actual_residual(adjoint: torch.Tensor) -> dict:
        rhs = upstream.to(adjoint.dtype)
        residual = rhs - poisson_operator(adjoint, mask.to(adjoint.dtype), r2, eta)
        absolute = torch.linalg.vector_norm(residual, dim=(-2, -1))
        rhs_norm = torch.linalg.vector_norm(rhs, dim=(-2, -1))
        return _residual_info({"absolute_residual": absolute,
            "relative_residual": absolute / rhs_norm.clamp_min(torch.finfo(adjoint.dtype).tiny)})

    reference = _pcg(upstream.double(), mask.double(), zero.double(), r2, eta,
                     reference_iterations, reference_tolerance)
    reference_norm = torch.linalg.vector_norm(reference).clamp_min(tiny)
    hole_reference_mass = (reference.abs() * hole).sum().clamp_min(tiny)
    reference_gradient = grad(reference)
    reference_p_vjp = r2 * reference_gradient
    gradient_hole = torch.cat((hole, hole), dim=1).double()
    gradient_norm = torch.linalg.vector_norm(reference_gradient).clamp_min(tiny)
    hole_gradient_norm = torch.linalg.vector_norm(reference_gradient * gradient_hole).clamp_min(tiny)
    rows = []
    for budget in BACKWARD_BUDGETS:
        adjoint = _pcg(upstream, mask, zero, r2, eta, budget, tolerance)
        difference = adjoint.double() - reference
        production_gradient = grad(adjoint)
        gradient_difference = production_gradient.double() - reference_gradient
        p_vjp_difference = (r2 * production_gradient).double() - reference_p_vjp
        rows.append({"iterations": budget, **actual_residual(adjoint),
            "hole_adjoint_l1_vs_reference": _hole_mean(difference.abs(), mask),
            "hole_adjoint_relative_l1_vs_reference": float((difference.abs() * hole).sum() / hole_reference_mass),
            "adjoint_relative_l2_vs_reference": float(torch.linalg.vector_norm(difference) / reference_norm),
            "gradient_relative_l2_vs_reference": float(torch.linalg.vector_norm(gradient_difference) / gradient_norm),
            "hole_gradient_relative_l2_vs_reference": float(torch.linalg.vector_norm(gradient_difference * gradient_hole) / hole_gradient_norm),
            "p_vjp_relative_l2_vs_reference": float(torch.linalg.vector_norm(p_vjp_difference) / torch.linalg.vector_norm(reference_p_vjp).clamp_min(tiny)),
            "hole_p_vjp_relative_l2_vs_reference": float(torch.linalg.vector_norm(p_vjp_difference * gradient_hole) / torch.linalg.vector_norm(reference_p_vjp * gradient_hole).clamp_min(tiny))})
    return {"loss": "hole-normalized RGB L1 of actual reference readout versus GT",
        "upstream_gradient_l1": float(upstream.abs().sum()),
        "upstream_gradient_l2": float(torch.linalg.vector_norm(upstream)),
        "gradient_is_zero": not bool(upstream.count_nonzero()),
        "requested_tolerance": tolerance, "budgets": rows,
        "reference": {"iterations": reference_iterations, "requested_tolerance": reference_tolerance,
                      "reference_dtype": "float64", "input_contract": "same FP32 upstream RHS and mask, postcast to FP64",
                      **actual_residual(reference)}}


def readout_sweep(outputs: Mapping, sample: Mapping, model: FeatureGuidedElasticaADMMNet,
                  reference_iterations: int, tolerance: float) -> tuple[dict, torch.Tensor]:
    """Use the actual forward's final state with the model's production PCG solver."""
    aux = outputs["aux"]
    arguments = (outputs["readout"]["u_before"], aux["p"], aux["lambda2"],
                 sample["masked"], sample["mask"], model.stage_hyper.r2, model.stage_hyper.eta)
    reference_tolerance = min(tolerance, 1e-10)
    reference_arguments = tuple(value.double() if torch.is_tensor(value) else value for value in arguments)
    reference, reference_info = solve_u_pcg(*reference_arguments, iterations=reference_iterations,
                                           tolerance=reference_tolerance)
    rows = []
    for budget in FORWARD_BUDGETS:
        pred, info = solve_u_pcg(*arguments, iterations=budget, tolerance=tolerance)
        rows.append({"iterations": budget, **_residual_info(info),
                     "hole_l1_vs_reference": _hole_mean((pred - reference).abs(), sample["mask"]),
                     "hole_l1_vs_gt": _hole_mean((pred - sample["gt"]).abs(), sample["mask"])})
    return {"readout_sweep": rows,
        "backward_sweep": backward_sweep(reference, sample["gt"], sample["mask"],
            model.stage_hyper.r2, model.stage_hyper.eta, reference_iterations, tolerance),
        "reference": {"iterations": reference_iterations, "requested_tolerance": reference_tolerance,
                      "reference_dtype": "float64", "input_contract": "same FP32 final model variables, postcast to FP64",
                      **_residual_info(reference_info)},
        "hole_l1_vs_gt_reference": _hole_mean((reference - sample["gt"]).abs(), sample["mask"]),
        "stages": [{name: float(value) for name, value in stage.items()} for stage in outputs["diagnostics"]]}, reference.cpu()


@torch.no_grad()
def run_preflight(cfg: Mapping, samples: list[dict], *, device: torch.device,
                  checkpoint: Path | None = None, reference_iterations: int = 1024) -> dict:
    if reference_iterations <= 512:
        raise ValueError("Reference iteration budget must exceed 512")
    diagnostic_cfg = copy.deepcopy(cfg)
    model_cfg = diagnostic_cfg.setdefault("model", {})
    if not model_cfg.get("use_unrolling", True):
        raise ValueError("Structure prior preflight requires model.use_unrolling=true")
    added_prior = not model_cfg.get("use_structure_prior", False)
    configured_rho = float(model_cfg.get("structure_rho", 0))
    model_cfg.update(use_structure_prior=True, structure_rho=max(configured_rho, 1.0), use_pcg_readout=False)
    _seed(int(cfg.get("seed", 42)))
    model = FeatureGuidedElasticaADMMNet.from_config(diagnostic_cfg).float()
    missing = []
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved.get("model", saved)
        incompatible = model.load_state_dict(state, strict=False)
        missing = incompatible.missing_keys
        if incompatible.unexpected_keys or any(not (added_prior and key.startswith("structure_prior.")) for key in missing):
            raise ValueError(f"Checkpoint does not match config: missing={missing}, unexpected={incompatible.unexpected_keys}")
    model = model.to(device).eval()
    edge_scale = float(cfg.get("loss", {}).get("edge_scale", .1))
    strong_threshold = float(cfg.get("loss", {}).get("orientation_threshold", .05))
    tolerance = float(cfg.get("model", {}).get("readout_tolerance", 1e-5))
    report = {"weights": str(checkpoint) if checkpoint else "random_initialization",
        "device": str(device), "dtype": "float32", "reference_dtype": "float64",
        "model_image_size": cfg["data"].get("image_size", 256),
        "diagnostic_model": {"added_prior_for_oracle": added_prior, "configured_rho": configured_rho,
            "missing_checkpoint_keys": missing, "readout": "production solver applied to actual final states",
            "comparison": "inference interventions on identical common weights; not retrained R0/R1/R2 controls"},
        "samples": [], "notes": ["Random weights do not establish inpainting quality or learned-prior effectiveness.",
            "Recommendations are provisional training-only numerical references; original config is never modified.",
            "Oracle uses GT only through the external prior_gradient diagnostic interface.",
            "References use the same FP32 model variables/RHS postcast to FP64, with the same production operator; the network remains FP32.",
            "FP32 second-order differences involve cancellation: recomputed residuals can be roundoff-sensitive even when FP64-referenced solution and VJP errors are small.",
            "Production FP32 residuals are reported fully; acceptance uses reliable FP64 references and solution/adjoint/VJP differences instead of a production-residual veto.",
            "Backward calibration covers representative normalized hole-L1 gradients only; it does not establish exact gradients for every training loss."]}
    strong_values, mean_magnitudes, critical_values, initial_thresholds = [], [], [], []
    baseline_predictions = []
    for index, cpu_sample in enumerate(samples):
        sample = {key: value.to(device) if torch.is_tensor(value) else value for key, value in cpu_sample.items()}
        g = grad(sample["gt"])
        norm = vector_norm(g)
        support = (norm > strong_threshold) & (sample["mask"] == 0)
        strong_values.append(norm[support].cpu())
        mean_norm = norm.mean(1, keepdim=True)
        mean_magnitudes.append(mean_norm[(sample["mask"] == 0) & (mean_norm > 0)].cpu())
        outputs = model(sample["masked"], sample["mask"], rho_scale=0, return_stage_states=True)
        first = outputs["stage_states"][0]
        critical = prior_activation_threshold(first["q"], first["c"], g, model.stage_hyper.r2)
        inactive = vector_norm(model.stage_hyper.r2 * first["q"]) <= first["c"]
        critical_values.append(critical[support & inactive].cpu())
        initial_thresholds.append((first["c"] / model.stage_hyper.r2)[support].cpu())
        disabled, pred = readout_sweep(outputs, sample, model, reference_iterations, tolerance)
        baseline_predictions.append(pred)
        report["samples"].append({"id": sample["id"], "mask_seed": sample["seed"],
            "mask_sha256": sample["mask_sha256"], "image_size": list(sample["gt"].shape[-2:]),
            "hole_ratio": float((1 - sample["mask"]).mean()), "strong_gt_vectors_in_hole": int(support.sum()),
            "fixed_scale_edge_mean_hole": _hole_mean(edge_strength(g, edge_scale), sample["mask"]),
            "injection_disabled": disabled})
        print(f"Disabled injection {index + 1}/{len(samples)}", flush=True)
        del outputs
    strong_stats = _quantiles(strong_values)
    mean_stats = _quantiles(mean_magnitudes)
    critical_stats = _quantiles(critical_values)
    rho_reference = 1.25 * critical_stats["p75"] if critical_stats["p75"] is not None else None
    if rho_reference is None and strong_stats["p50"] is not None:
        rho_reference = 1.25 * (model.stage_hyper.a + model.stage_hyper.r1) / strong_stats["p50"]
    diagnostic_rho = max(configured_rho, rho_reference or model.stage_hyper.r2)
    model.structure_rho = diagnostic_rho
    for index, cpu_sample in enumerate(samples):
        sample = {key: value.to(device) if torch.is_tensor(value) else value for key, value in cpu_sample.items()}
        outputs = model(sample["masked"], sample["mask"], prior_gradient=grad(sample["gt"]), rho_scale=1)
        oracle, pred = readout_sweep(outputs, sample, model, reference_iterations, tolerance)
        row = report["samples"][index]
        row["oracle_gt_prior"] = oracle
        row["oracle_minus_disabled_hole_l1"] = oracle["hole_l1_vs_gt_reference"] - row["injection_disabled"]["hole_l1_vs_gt_reference"]
        row["oracle_vs_disabled_prediction_hole_l1"] = _hole_mean((pred - baseline_predictions[index]).abs(), cpu_sample["mask"])
        print(f"Oracle GT prior {index + 1}/{len(samples)}", flush=True)
    scenarios = [row[name] for row in report["samples"] for name in ("injection_disabled", "oracle_gt_prior")]
    reference_max = max(case["reference"]["relative_residual_max"] for case in scenarios)
    reference_reliable = reference_max <= 1e-8
    candidates = [budget for budget in FORWARD_BUDGETS if reference_reliable and all(
        entry["hole_l1_vs_reference"] <= 1e-3
        for case in scenarios for entry in case["readout_sweep"] if entry["iterations"] == budget)]
    backward_cases = [case["backward_sweep"] for case in scenarios if not case["backward_sweep"]["gradient_is_zero"]]
    backward_reference_max = max((case["reference"]["relative_residual_max"] for case in backward_cases), default=None)
    backward_reliable = backward_reference_max is not None and backward_reference_max <= 1e-8
    backward_candidates = [budget for budget in BACKWARD_BUDGETS if backward_reliable and all(
        entry["hole_adjoint_relative_l1_vs_reference"] <= 1e-3
        and entry["adjoint_relative_l2_vs_reference"] <= 1e-3
        and entry["gradient_relative_l2_vs_reference"] <= 1e-3
        and entry["hole_gradient_relative_l2_vs_reference"] <= 1e-3
        and entry["p_vjp_relative_l2_vs_reference"] <= 1e-3
        and entry["hole_p_vjp_relative_l2_vs_reference"] <= 1e-3
        for case in backward_cases for entry in case["budgets"] if entry["iterations"] == budget)]
    report.update(gt_gradient_statistics={"strong_threshold_raw_vector_magnitude": strong_threshold,
        "strong_hole_rgb_vector_norm": strong_stats, "nonzero_hole_mean_rgb_vector_norm": mean_stats,
        "initial_c_over_r2_on_strong_gt": _quantiles(initial_thresholds),
        "rho_critical_initially_inactive_strong_gt": critical_stats},
        recommendations={"edge_scale_configured": edge_scale,
            "edge_scale_reference": mean_stats["p75"] / math.log(2) if mean_stats["p75"] is not None else None,
            "edge_scale_reference_rule": "training p75 of nonzero mean-RGB gradient mapped to soft edge 0.5",
            "rho_reference": rho_reference, "oracle_diagnostic_rho": diagnostic_rho,
            "rho_reference_rule": "1.25 * p75 of actual first-stage inactive-vector critical rho; flat-region median fallback",
            "readout_iterations_reference": min(candidates) if candidates else None,
            "readout_reference_reliable": reference_reliable, "readout_reference_actual_residual_max": reference_max,
            "reference_relative_residual_goal": 1e-8,
            "production_residual_is_acceptance_gate": False, "readout_hole_l1_difference_goal": 1e-3,
            "backward_iterations_reference": min(backward_candidates) if backward_candidates else None,
            "backward_nonzero_rhs_cases": len(backward_cases),
            "backward_reference_reliable": backward_reliable,
            "backward_reference_actual_residual_max": backward_reference_max,
            "backward_relative_adjoint_difference_goal": 1e-3,
            "backward_relative_vjp_difference_goal": 1e-3})
    if not strong_stats["count"]:
        report["notes"].append("No strong GT edges in holes: rho cannot be calibrated; the oracle rho is a diagnostic fallback.")
    if not reference_reliable:
        report["notes"].append("High-budget reference residual is too large: no readout budget is recommended.")
    if not backward_reliable:
        report["notes"].append("Backward reference lacks nonzero support or adequate actual residual: no backward budget is recommended.")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--synthetic", action="store_true", help="Explicit synthetic self-check; not dataset calibration")
    parser.add_argument("--synthetic-size", choices=(32, 64), type=int, default=32)
    parser.add_argument("--reference-iterations", type=int, default=1024,
                        help="Forward reference must exceed 512; backward reference uses at least 1024")
    args = parser.parse_args(argv)
    try:
        config_path = _resolve(args.config)
        cfg = load_config(config_path)
        output = Path(args.output).expanduser().resolve()
        checkpoint = _resolve(args.checkpoint) if args.checkpoint else None
        if output == config_path or output == checkpoint:
            raise ValueError("Report output must not overwrite the config or checkpoint")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
        if device.type == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 4))
        samples = load_samples(cfg, args.samples, seed=int(cfg.get("seed", 42)), synthetic=args.synthetic,
                               synthetic_size=args.synthetic_size, config_path=config_path)
        report = run_preflight(cfg, samples, device=device, checkpoint=checkpoint,
                               reference_iterations=args.reference_iterations)
        report.update(config=str(config_path), requested_samples=args.samples, actual_samples=len(samples),
            data_source="synthetic_diagnostic_only" if args.synthetic else "train_list",
            calibration_eligible=not args.synthetic)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(output), "recommendations": report["recommendations"]}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"preflight error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
