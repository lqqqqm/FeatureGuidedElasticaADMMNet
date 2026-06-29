from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..models.operators import grad


class OptionalLPIPS:
    def __init__(self, device: torch.device):
        self.ready = False
        self.model = None
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="alex").to(device)
            self.model.eval()
            self.ready = True
        except Exception:
            self.ready = False

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.ready or self.model is None:
            return None
        return self.model(pred, target).mean()



def composite_hole(pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    M3 = M.repeat(1, 3, 1, 1)
    return (1.0 - M3) * pred + M3 * gt



def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, gt)
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-8))



def psnr_hole(pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    hole = (1.0 - M).repeat(1, 3, 1, 1)
    mse = ((pred - gt).pow(2) * hole).sum() / hole.sum().clamp_min(1.0)
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-8))


def gradient_l1(pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
    diff = (grad(pred) - grad(gt)).abs()
    if M is None:
        return diff.mean()
    mask = (1.0 - M).repeat(1, diff.shape[1], 1, 1)
    return (diff * mask).sum() / mask.sum().clamp_min(1.0)


def _edge_map(x: torch.Tensor, threshold: float = 0.15) -> torch.Tensor:
    g = grad(x)
    c = x.shape[1]
    gx, gy = g[:, :c], g[:, c:]
    mag = (gx.square() + gy.square() + 1e-6).sqrt().mean(dim=1, keepdim=True)
    mag = mag / mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (mag > threshold).float()


def edge_f1(pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
    pred_edge = _edge_map(pred)
    gt_edge = _edge_map(gt)
    mask = torch.ones_like(pred_edge) if M is None else (1.0 - M)
    tp = (pred_edge * gt_edge * mask).sum()
    fp = (pred_edge * (1.0 - gt_edge) * mask).sum()
    fn = ((1.0 - pred_edge) * gt_edge * mask).sum()
    return 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-6)


def boundary_consistency(pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    hole = 1.0 - M
    boundary = F.max_pool2d(hole, kernel_size=3, stride=1, padding=1) - hole
    boundary3 = boundary.repeat(1, pred.shape[1], 1, 1)
    return ((pred - gt).abs() * boundary3).sum() / boundary3.sum().clamp_min(1.0)



def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device=None, dtype=None):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
    return kernel_2d



def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    c1 = (0.01 * 2.0) ** 2
    c2 = (0.03 * 2.0) ** 2
    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels, device=pred.device, dtype=pred.dtype)
    mu_x = F.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(gt, kernel, padding=window_size // 2, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(gt * gt, kernel, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * gt, kernel, padding=window_size // 2, groups=channels) - mu_xy
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return score.mean()



def evaluate_batch(pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor, lpips_metric: OptionalLPIPS | None = None) -> Dict[str, float]:
    items = evaluate_per_image(pred, gt, M, lpips_metric)
    if not items:
        return {}

    out: Dict[str, float] = {}
    metric_names = [key for key in items[0].keys() if key not in {"index", "hole_ratio"}]
    for name in metric_names:
        values = [float(item[name]) for item in items if name in item]
        if values:
            out[name] = sum(values) / len(values)
    return out


def evaluate_per_image(
    pred: torch.Tensor,
    gt: torch.Tensor,
    M: torch.Tensor,
    lpips_metric: OptionalLPIPS | None = None,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    comp = composite_hole(pred, gt, M)
    batch_size = pred.shape[0]
    for idx in range(batch_size):
        pred_i = pred[idx : idx + 1]
        gt_i = gt[idx : idx + 1]
        mask_i = M[idx : idx + 1]
        comp_i = comp[idx : idx + 1]

        item: Dict[str, float] = {
            "index": float(idx),
            "hole_ratio": float((1.0 - mask_i).mean()),
            "psnr": float(psnr(pred_i, gt_i)),
            "psnr_hole": float(psnr_hole(pred_i, gt_i, mask_i)),
            "ssim": float(ssim(pred_i, gt_i)),
            "ssim_hole": float(ssim(comp_i, gt_i)),
            "edge_f1": float(edge_f1(pred_i, gt_i, mask_i)),
            "gradient_l1": float(gradient_l1(pred_i, gt_i, mask_i)),
            "boundary_consistency": float(boundary_consistency(pred_i, gt_i, mask_i)),
        }
        if lpips_metric is not None:
            lpips_whole = lpips_metric(pred_i, gt_i)
            lpips_h = lpips_metric(comp_i, gt_i)
            if lpips_whole is not None:
                item["lpips"] = float(lpips_whole)
            if lpips_h is not None:
                item["lpips_hole"] = float(lpips_h)
        results.append(item)
    return results
