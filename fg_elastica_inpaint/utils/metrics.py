from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

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


class FrechetInceptionDistance:
    """Lazy wrapper around TorchMetrics' standard Inception-v3 FID metric.

    Images in this project use the [-1, 1] range.  TorchMetrics receives the
    corresponding [0, 1] floating-point tensors via ``normalize=True``.
    """

    def __init__(self, device: torch.device):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance as TorchMetricsFID
        except Exception as exc:
            raise RuntimeError(
                "FID requires the optional dependency 'torchmetrics[image]'. "
                "Install requirements_optional.txt before setting eval.compute_fid=true."
            ) from exc

        try:
            self.metric = TorchMetricsFID(feature=2048, normalize=True).to(device)
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize the Inception-v3 FID metric. Ensure its pretrained weights "
                "are cached or allow Kaggle Internet access for the first run."
            ) from exc

    @staticmethod
    def _to_unit_range(x: torch.Tensor) -> torch.Tensor:
        return ((x.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def update(self, generated: torch.Tensor, real: torch.Tensor) -> None:
        self.metric.update(self._to_unit_range(real), real=True)
        self.metric.update(self._to_unit_range(generated), real=False)

    @torch.no_grad()
    def compute(self) -> float:
        return float(self.metric.compute())



MASK_RATIO_BUCKETS = [
    ("0-10%", 0.0, 0.1),
    ("10-20%", 0.1, 0.2),
    ("20-30%", 0.2, 0.3),
    ("30-40%", 0.3, 0.4),
    ("40-50%", 0.4, 0.5),
    ("50-60%", 0.5, 0.6),
    ("60%+", 0.6, 1.0 + 1e-6),
]


def composite_completed(pred: torch.Tensor, gt: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
    known_mask3 = known_mask.repeat(1, pred.shape[1], 1, 1)
    hole_mask3 = 1.0 - known_mask3
    return hole_mask3 * pred + known_mask3 * gt


def composite_hole(pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias. M=1 is known, M=0 is hole."""
    return composite_completed(pred, gt, M)



def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, gt)
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
    hole_mask = 1.0 - M
    eroded_hole = -F.max_pool2d(-hole_mask, kernel_size=5, stride=1, padding=2)
    boundary = (hole_mask - eroded_hole).clamp(0.0, 1.0)
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
    completed = composite_completed(pred, gt, M)
    batch_size = pred.shape[0]
    for idx in range(batch_size):
        gt_i = gt[idx : idx + 1]
        known_mask_i = M[idx : idx + 1]
        completed_i = completed[idx : idx + 1]

        item: Dict[str, float] = {
            "index": float(idx),
            "hole_ratio": float((1.0 - known_mask_i).mean()),
            "psnr": float(psnr(completed_i, gt_i)),
            "ssim": float(ssim(completed_i, gt_i)),
            "l1": float(F.l1_loss(completed_i, gt_i)),
            "edge_f1": float(edge_f1(completed_i, gt_i)),
            "gradient_l1": float(gradient_l1(completed_i, gt_i)),
            "boundary_consistency": float(boundary_consistency(completed_i, gt_i, known_mask_i)),
        }
        if lpips_metric is not None:
            lpips_whole = lpips_metric(completed_i, gt_i)
            if lpips_whole is not None:
                item["lpips"] = float(lpips_whole)
        results.append(item)
    return results


def mask_ratio_bucket(mask_ratio: float) -> str:
    for name, low, high in MASK_RATIO_BUCKETS:
        if low <= mask_ratio < high:
            return name
    return "60%+"


def metric_names_from_items(items: Iterable[Dict[str, float]]) -> List[str]:
    names: List[str] = []
    for item in items:
        for key in item:
            if key in {"index", "hole_ratio"} or key in names:
                continue
            names.append(key)
    return names


def summarize_metric_items(items: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for name in metric_names_from_items(items):
        values = [float(item[name]) for item in items if name in item and math.isfinite(float(item[name]))]
        if values:
            summary[name] = sum(values) / len(values)
    return summary


def summarize_bucket_metrics(items: List[Dict[str, float]]) -> List[Dict[str, float | str]]:
    metric_names = metric_names_from_items(items)
    rows: List[Dict[str, float | str]] = []
    for bucket, _, _ in MASK_RATIO_BUCKETS:
        bucket_items = [item for item in items if mask_ratio_bucket(float(item["hole_ratio"])) == bucket]
        row: Dict[str, float | str] = {
            "bucket": bucket,
            "num_samples": len(bucket_items),
            "avg_mask_ratio": float("nan"),
        }
        if bucket_items:
            row["avg_mask_ratio"] = sum(float(item["hole_ratio"]) for item in bucket_items) / len(bucket_items)
        for name in metric_names:
            values = [float(item[name]) for item in bucket_items if name in item and math.isfinite(float(item[name]))]
            row[name] = sum(values) / len(values) if values else float("nan")
        rows.append(row)
    return rows


def validation_sanity(
    items: List[Dict[str, float]],
    completed_min: float,
    completed_max: float,
    min_bucket_samples: int = 5,
) -> List[str]:
    if not items:
        return ["No evaluation samples were processed."]
    lines = []
    avg_mask_ratio = sum(float(item["hole_ratio"]) for item in items) / len(items)
    lines.append(f"Average mask ratio: {avg_mask_ratio:.4f}")
    lines.append(f"Completed image range: min={completed_min:.4f}, max={completed_max:.4f}")
    for row in summarize_bucket_metrics(items):
        count = int(row["num_samples"])
        lines.append(f"Bucket {row['bucket']}: {count} sample(s)")
        if 0 < count < min_bucket_samples:
            lines.append(f"WARNING: bucket {row['bucket']} has only {count} sample(s); metrics may be unstable.")
    for metric_name in ["psnr", "ssim", "lpips"]:
        bad_count = sum(
            1
            for item in items
            if metric_name in item and not math.isfinite(float(item[metric_name]))
        )
        if bad_count:
            lines.append(f"WARNING: {metric_name} produced {bad_count} non-finite value(s).")
    return lines
