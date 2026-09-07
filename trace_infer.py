from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
import torch

from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.models.operators import (
    div,
    dot_mp,
    grad,
    laplace,
    normalize_vec,
    vector_norm,
)
from fg_elastica_inpaint.models.unfolding import (
    solve_m_proj,
    solve_n_gd,
    solve_p_shrink,
    solve_u_gd,
)
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.image import pil_to_tensor, tensor_to_pil


def load_rgb(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return pil_to_tensor(img, value_range="minus_one_to_one").unsqueeze(0)


def load_mask(path: str, size: int, invert: bool = False) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    if invert:
        arr = 1.0 - arr
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def save_rgb(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(t).save(path)


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    lo = x.amin()
    hi = x.amax()
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def save_gray_map(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.mean(dim=0)
    arr = (_normalize01(x).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def save_signed_map(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.mean(dim=0)
    vmax = x.abs().amax().clamp_min(1e-8)
    red = (x.clamp_min(0.0) / vmax)
    blue = ((-x).clamp_min(0.0) / vmax)
    green = 1.0 - (red + blue).clamp(0.0, 1.0)
    rgb = torch.stack([red, green, blue], dim=-1)
    arr = (rgb.numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def save_channel_grid(x: torch.Tensor, path: Path, max_channels: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    channels = min(int(x.shape[0]), max_channels)
    maps = [_normalize01(x[i]) for i in range(channels)]
    h, w = maps[0].shape
    cols = 4
    rows = (channels + cols - 1) // cols
    grid = torch.zeros(rows * h, cols * w)
    for idx, item in enumerate(maps):
        r = idx // cols
        c = idx % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = item
    arr = (grid.numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def vector_magnitude(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] % 2 == 0:
        return vector_norm(x)
    return x.abs()


def tensor_stats(t: Optional[torch.Tensor]) -> Optional[Dict[str, object]]:
    if t is None:
        return None
    x = t.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "l2": float(torch.linalg.vector_norm(x)),
    }


def add_tensor(trace: Dict[str, torch.Tensor], stats: Dict[str, object], name: str, value: Optional[torch.Tensor]) -> None:
    if value is None:
        stats[name] = None
        return
    trace[name] = value.detach().cpu()
    stats[name] = tensor_stats(value)


@torch.no_grad()
def trace_forward(model: FeatureGuidedElasticaADMMNet, gt: torch.Tensor, M: torch.Tensor):
    I_m = gt * M
    trace: Dict[str, torch.Tensor] = {}
    stats: Dict[str, object] = {}

    add_tensor(trace, stats, "input", gt)
    add_tensor(trace, stats, "mask", M)
    add_tensor(trace, stats, "masked", I_m)

    x = torch.cat([I_m, M], dim=1)
    F1, F2, F3 = model.backbone(x)
    add_tensor(trace, stats, "F1", F1)
    add_tensor(trace, stats, "F2", F2)
    add_tensor(trace, stats, "F3", F3)

    u = I_m
    p = grad(u)
    m = normalize_vec(p, model.eps)
    n = normalize_vec(p, model.eps)
    b, _, h, w = I_m.shape
    lambda1 = torch.zeros(b, 3, h, w, device=I_m.device, dtype=I_m.dtype)
    lambda2 = torch.zeros(b, 6, h, w, device=I_m.device, dtype=I_m.dtype)
    lambda4 = torch.zeros(b, 6, h, w, device=I_m.device, dtype=I_m.dtype)

    add_tensor(trace, stats, "u_0", u)
    add_tensor(trace, stats, "p_0", p)
    add_tensor(trace, stats, "m_0", m)
    add_tensor(trace, stats, "n_0", n)
    add_tensor(trace, stats, "lambda1_0", lambda1)
    add_tensor(trace, stats, "lambda2_0", lambda2)
    add_tensor(trace, stats, "lambda4_0", lambda4)

    stage = model.stage
    F_ms = stage.fuse(F1, F2, F3)
    add_tensor(trace, stats, "F_ms", F_ms)

    stage_preds = []
    for k in range(model.K):
        prefix = f"stage_{k + 1}"
        hole3 = (1.0 - M).repeat(1, u.shape[1], 1, 1)
        hole6 = (1.0 - M).repeat(1, p.shape[1], 1, 1)

        structure_edge = None
        structure_orient = None
        if stage.use_structure_head:
            structure_edge, structure_orient = stage.structure_head(F_ms, stage.hyper.eps)
            structure_orient = hole6 * structure_orient

        if stage.use_unrolling:
            u_tilde = solve_u_gd(
                u_init=u,
                p=p,
                lambda2=lambda2,
                I_m=I_m,
                M=M,
                r2=stage.hyper.r2,
                eta=stage.hyper.eta,
                Tu=stage.hyper.Tu,
                tau_u=stage.hyper.tau_u,
            )
        else:
            u_tilde = u

        u_stationarity = -stage.hyper.r2 * laplace(u_tilde) + stage.hyper.eta * M * u_tilde
        u_stationarity = u_stationarity - (
            stage.hyper.eta * M * I_m - div(stage.hyper.r2 * p + lambda2)
        )
        base_u = -u_stationarity
        F_u = stage.psi_u(
            semantic=F_ms,
            variable=u_tilde,
            mask=M,
            residual=base_u,
        )
        du = None
        if stage.enable_u_correction:
            du = stage.phi_u(
                variable=u_tilde,
                adapted_feature=F_u,
                mask=M,
                base_correction=base_u,
            )
            u_new = u_tilde + model._alpha(model.alpha_u, k, "u") * hole3 * du
        else:
            u_new = u_tilde

        if stage.use_unrolling:
            p_tilde = solve_p_shrink(
                u=u_new,
                m=m,
                n=n,
                lambda1=lambda1,
                lambda2=lambda2,
                a=stage.hyper.a,
                b=stage.hyper.b,
                r1=stage.hyper.r1,
                r2=stage.hyper.r2,
            )
        else:
            p_tilde = p

        base_p = grad(u_new) - p_tilde
        F_p = stage.psi_p(
            semantic=F_ms,
            variable=p_tilde,
            mask=M,
            residual=base_p,
        )
        dp = None
        if stage.enable_p_correction:
            dp = stage.phi_p(
                variable=p_tilde,
                adapted_feature=F_p,
                mask=M,
                base_correction=base_p,
            )
            p_new = p_tilde + model._alpha(model.alpha_p, k, "p") * hole6 * dp
        else:
            p_new = p_tilde

        if stage.use_unrolling:
            m_new = solve_m_proj(
                p=p_new,
                n=n,
                lambda1=lambda1,
                lambda4=lambda4,
                r1=stage.hyper.r1,
                r4=stage.hyper.r4,
            )
            n_tilde = solve_n_gd(
                n_init=n,
                p=p_new,
                m=m_new,
                lambda4=lambda4,
                r4=stage.hyper.r4,
                b=stage.hyper.b,
                Tn=stage.hyper.Tn,
                tau_n=stage.hyper.tau_n,
            )
        else:
            m_new = normalize_vec(p_new, stage.hyper.eps)
            n_tilde = n

        base_n = m_new - n_tilde
        F_n = stage.psi_n(
            semantic=F_ms,
            variable=n_tilde,
            mask=M,
            residual=base_n,
        )
        dn = None
        if stage.enable_n_correction:
            dn = stage.phi_n(
                variable=n_tilde,
                adapted_feature=F_n,
                mask=M,
                base_correction=base_n,
            )
            n_new = normalize_vec(n_tilde + model._alpha(model.alpha_n, k, "n") * hole6 * dn, stage.hyper.eps)
        else:
            n_new = n_tilde

        if stage.use_unrolling:
            lambda1_new = lambda1 + stage.hyper.r1 * (vector_norm(p_new) - dot_mp(m_new, p_new))
            lambda2_new = lambda2 + stage.hyper.r2 * (p_new - grad(u_new))
            lambda4_new = lambda4 + stage.hyper.r4 * (n_new - m_new)
        else:
            lambda1_new = lambda1
            lambda2_new = lambda2
            lambda4_new = lambda4

        add_tensor(trace, stats, f"{prefix}.u_tilde", u_tilde)
        add_tensor(trace, stats, f"{prefix}.F_u", F_u)
        add_tensor(trace, stats, f"{prefix}.du", du)
        add_tensor(trace, stats, f"{prefix}.u", u_new)
        add_tensor(trace, stats, f"{prefix}.p_tilde", p_tilde)
        add_tensor(trace, stats, f"{prefix}.F_p", F_p)
        add_tensor(trace, stats, f"{prefix}.dp", dp)
        add_tensor(trace, stats, f"{prefix}.p", p_new)
        add_tensor(trace, stats, f"{prefix}.m", m_new)
        add_tensor(trace, stats, f"{prefix}.n_tilde", n_tilde)
        add_tensor(trace, stats, f"{prefix}.F_n", F_n)
        add_tensor(trace, stats, f"{prefix}.dn", dn)
        add_tensor(trace, stats, f"{prefix}.n", n_new)
        add_tensor(trace, stats, f"{prefix}.lambda1", lambda1_new)
        add_tensor(trace, stats, f"{prefix}.lambda2", lambda2_new)
        add_tensor(trace, stats, f"{prefix}.lambda4", lambda4_new)
        add_tensor(trace, stats, f"{prefix}.structure_edge", structure_edge)
        add_tensor(trace, stats, f"{prefix}.structure_orient", structure_orient)

        u, p, m, n = u_new, p_new, m_new, n_new
        lambda1, lambda2, lambda4 = lambda1_new, lambda2_new, lambda4_new
        stage_preds.append(u)

    pred = u
    comp = M * I_m + (1.0 - M) * pred
    add_tensor(trace, stats, "pred", pred)
    add_tensor(trace, stats, "comp", comp)
    for idx, item in enumerate(stage_preds, start=1):
        add_tensor(trace, stats, f"stage_pred_{idx}", item)

    return trace, stats


def write_visualizations(trace: Dict[str, torch.Tensor], out_dir: Path) -> None:
    images_dir = out_dir / "images"
    features_dir = out_dir / "features"
    vectors_dir = out_dir / "vectors"
    lambdas_dir = out_dir / "lambdas"

    for name in ["input", "mask", "masked", "u_0", "pred", "comp"]:
        if name in trace:
            if name == "mask":
                save_gray_map(trace[name], images_dir / f"{name}.png")
            else:
                save_rgb(trace[name], images_dir / f"{name}.png")

    stage_idx = 1
    while f"stage_{stage_idx}.u" in trace:
        save_rgb(trace[f"stage_{stage_idx}.u_tilde"], images_dir / f"stage_{stage_idx}_u_tilde.png")
        save_rgb(trace[f"stage_{stage_idx}.u"], images_dir / f"stage_{stage_idx}_u.png")
        if f"stage_{stage_idx}.du" in trace:
            save_signed_map(trace[f"stage_{stage_idx}.du"], images_dir / f"stage_{stage_idx}_du_signed.png")
        stage_idx += 1

    for name in ["F1", "F2", "F3", "F_ms"]:
        if name in trace:
            save_gray_map(trace[name].abs().mean(dim=1, keepdim=True), features_dir / f"{name}_mean_abs.png")
            save_channel_grid(trace[name], features_dir / f"{name}_channels_0_15.png")

    stage_idx = 1
    while f"stage_{stage_idx}.F_u" in trace:
        for suffix in ["F_u", "F_p", "F_n"]:
            key = f"stage_{stage_idx}.{suffix}"
            if key in trace:
                save_gray_map(trace[key].abs().mean(dim=1, keepdim=True), features_dir / f"stage_{stage_idx}_{suffix}_mean_abs.png")
                save_channel_grid(trace[key], features_dir / f"stage_{stage_idx}_{suffix}_channels_0_15.png")

        for suffix in ["p_tilde", "p", "m", "n_tilde", "n"]:
            key = f"stage_{stage_idx}.{suffix}"
            if key in trace:
                save_gray_map(vector_magnitude(trace[key]), vectors_dir / f"stage_{stage_idx}_{suffix}_magnitude.png")

        for suffix in ["lambda1", "lambda2", "lambda4"]:
            key = f"stage_{stage_idx}.{suffix}"
            if key in trace:
                save_signed_map(trace[key], lambdas_dir / f"stage_{stage_idx}_{suffix}_signed.png")
        stage_idx += 1


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--invert_mask", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = int(cfg["data"].get("image_size", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    model = FeatureGuidedElasticaADMMNet.from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gt = load_rgb(args.image, size).to(device)
    M = load_mask(args.mask, size, invert=args.invert_mask).to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace, stats = trace_forward(model, gt, M)
    write_visualizations(trace, out_dir)
    torch.save(trace, out_dir / "raw_tensors.pt")
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved trace outputs to {out_dir.resolve()}")
    print(f"Saved {len(trace)} tensors and stats for {len(stats)} entries.")


if __name__ == "__main__":
    main()
