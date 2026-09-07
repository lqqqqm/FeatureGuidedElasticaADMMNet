from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from ..models.operators import grad
from .structure import edge_strength


class OptionalLPIPS:
    def __init__(self, device: torch.device):
        self.ready = False
        self.model = None
        self.unavailable_reason: str | None = None
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="alex").to(device)
            self.model.eval()
            self.ready = True
        except Exception as exc:
            self.ready = False
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"

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


def _edge_map(x: torch.Tensor, threshold: float = 0.5, scale: float = 0.1) -> torch.Tensor:
    """Threshold the shared fixed-scale strength; never normalize per image."""
    return (edge_strength(grad(x), scale=scale) > threshold).float()


def _region_mean(values: torch.Tensor, region: torch.Tensor, empty_value: float = 0.0) -> torch.Tensor:
    weights = region.expand_as(values)
    count = weights.sum()
    value = (values * weights).sum() / count.clamp_min(1.0)
    return torch.where(count > 0, value, value.new_tensor(empty_value))


def _edge_scores(
    pred_edge: torch.Tensor, gt_edge: torch.Tensor, region: torch.Tensor, tolerance: int = 1,
) -> Dict[str, torch.Tensor]:
    """Binary edge metrics, optionally matched within Chebyshev pixel radius.

    Matching is restricted to edges in the evaluation region before dilation.
    Both-empty edge sets score one; a missing nonempty set scores F1=0.
    Tolerance matching uses independent precision/recall matches (many-to-one).
    """
    pred_edge = pred_edge.float() * region
    gt_edge = gt_edge.float() * region
    predicted, actual = pred_edge.sum(), gt_edge.sum()
    tp = (pred_edge * gt_edge).sum()
    precision = torch.where(predicted > 0, tp / predicted.clamp_min(1), tp.new_ones(()))
    recall = torch.where(actual > 0, tp / actual.clamp_min(1), tp.new_ones(()))
    f1 = torch.where(predicted + actual > 0, 2 * tp / (predicted + actual).clamp_min(1), tp.new_ones(()))
    if tolerance:
        kernel = 2 * tolerance + 1
        near_gt = F.max_pool2d(gt_edge, kernel, stride=1, padding=tolerance)
        near_pred = F.max_pool2d(pred_edge, kernel, stride=1, padding=tolerance)
        matched_pred = (pred_edge * near_gt).sum()
        matched_gt = (gt_edge * near_pred).sum()
        tol_precision = torch.where(predicted > 0, matched_pred / predicted.clamp_min(1), tp.new_ones(()))
        tol_recall = torch.where(actual > 0, matched_gt / actual.clamp_min(1), tp.new_ones(()))
        tol_f1 = 2 * tol_precision * tol_recall / (tol_precision + tol_recall).clamp_min(1e-12)
    else:
        tol_precision, tol_recall, tol_f1 = precision, recall, f1
    return {"precision": precision, "recall": recall, "f1": f1,
            "tolerance_precision": tol_precision, "tolerance_recall": tol_recall,
            "tolerance_f1": tol_f1, "predicted_count": predicted, "target_count": actual}


def edge_f1(
    pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor | None = None,
    *, edge_scale: float = 0.1, threshold: float = 0.5,
) -> torch.Tensor:
    pred_edge = _edge_map(pred, threshold, edge_scale)
    gt_edge = _edge_map(gt, threshold, edge_scale)
    mask = torch.ones_like(pred_edge) if M is None else (1.0 - M)
    return _edge_scores(pred_edge, gt_edge, mask, tolerance=0)["f1"]


def _orientation_metrics(
    gradient: torch.Tensor, target: torch.Tensor, region: torch.Tensor, threshold: float,
) -> Dict[str, torch.Tensor]:
    """Mean signed-vector angle in radians, per RGB vector on strong GT only.

    A reversed gradient incurs pi; a missing prediction incurs pi/2. Report
    the support count so an empty GT region's zero error is not mistaken for
    a measured direction. Orientation is not taken modulo pi.
    """
    px, py = gradient.chunk(2, dim=1)
    tx, ty = target.chunk(2, dim=1)
    magnitude = torch.linalg.vector_norm(torch.stack((tx, ty)), dim=0)
    support = (magnitude > threshold).float() * region
    dot = px * tx + py * ty
    cross = px * ty - py * tx
    angle = torch.atan2(cross.abs(), dot)
    predicted_magnitude = torch.linalg.vector_norm(torch.stack((px, py)), dim=0)
    angle = torch.where(predicted_magnitude > 1e-12, angle, angle.new_tensor(math.pi / 2))
    return {"error": _region_mean(angle, support), "count": support.sum()}


def _hole_regions(M: torch.Tensor, boundary_width: int) -> Dict[str, torch.Tensor]:
    """Split holes by distance to a known pixel, with no SciPy dependency.

    Boundary includes Chebyshev distances 1..boundary_width; inner includes
    larger distances. The image exterior is not treated as a known pixel,
    so a fully missing image has no boundary and is entirely inner.
    """
    hole = 1.0 - M
    if boundary_width == 0:
        boundary = torch.zeros_like(hole)
    else:
        near_known = F.max_pool2d(M, 2 * boundary_width + 1, stride=1, padding=boundary_width)
        boundary = hole * near_known
    return {"hole": hole, "hole_boundary": boundary, "hole_inner": hole - boundary}


def boundary_consistency(pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    boundary = _hole_regions(M, boundary_width=2)["hole_boundary"]
    return _region_mean((pred - gt).abs(), boundary)



def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device=None, dtype=None):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
    return kernel_2d



def ssim_map(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Return the usual local SSIM map, before any region averaging."""
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
    return score


def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    return ssim_map(pred, gt, window_size, sigma).mean()



def evaluate_batch(
    pred: torch.Tensor, gt: torch.Tensor, M: torch.Tensor,
    lpips_metric: OptionalLPIPS | None = None, **metric_options,
) -> Dict[str, float]:
    items = evaluate_per_image(pred, gt, M, lpips_metric, **metric_options)
    return summarize_metric_items(items)


def _validate_metric_options(edge_scale, edge_threshold, edge_tolerance, orientation_threshold, boundary_width=0):
    if not math.isfinite(edge_scale) or edge_scale <= 0:
        raise ValueError("edge_scale must be finite and positive.")
    if not math.isfinite(edge_threshold) or not 0 <= edge_threshold < 1:
        raise ValueError("edge_threshold must lie in [0, 1).")
    if not math.isfinite(orientation_threshold) or orientation_threshold < 0:
        raise ValueError("orientation_threshold must be finite and nonnegative.")
    for name, value in (("edge_tolerance", edge_tolerance), ("boundary_width", boundary_width)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer number of pixels.")


def _validate_image_mask(gt: torch.Tensor, M: torch.Tensor) -> None:
    if gt.ndim != 4 or gt.shape[1] != 3 or min(gt.shape[2:]) < 1:
        raise ValueError("Metrics expect RGB images [B,3,H,W] with nonempty spatial dimensions.")
    if M.shape != (gt.shape[0], 1, gt.shape[2], gt.shape[3]):
        raise ValueError("Known mask must be [B,1,H,W] matching the image.")
    _require_finite(gt=gt, M=M)
    if not bool(torch.all((M == 0) | (M == 1))):
        raise ValueError("Metrics expect a binary known mask (1=known, 0=hole).")


def _require_finite(**tensors: torch.Tensor) -> None:
    """Reject failed predictions before reductions can hide them in summaries."""
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Evaluation input {name} contains NaN or infinity.")


@torch.no_grad()
def evaluate_per_image(
    pred: torch.Tensor,
    gt: torch.Tensor,
    M: torch.Tensor,
    lpips_metric: OptionalLPIPS | None = None,
    *,
    edge_scale: float = 0.1,
    edge_threshold: float = 0.5,
    edge_tolerance: int = 1,
    orientation_threshold: float = 0.05,
    boundary_width: int = 4,
) -> List[Dict[str, float]]:
    """Evaluate the completed RGB image, retaining existing whole-image keys.

    ``hole_*`` metrics average maps only at missing pixels, using the actual
    RGB composite and its gradients, never a zero-masked LPIPS input. SSIM is
    computed with the original local windows, then averaged in each region.
    Edge strength uses a fixed scale shared with structure supervision.

    Boundary/inner split uses Chebyshev distance to known pixels, in pixels
    at the evaluated resolution. Empty regions have zero errors, SSIM/F1=1
    and PSNR=86.0206 (the existing 1e-8 MSE floor); accompanying pixel/edge
    counts expose lack of support. These placeholders are excluded by the
    summary helpers. Orientation is in radians per RGB vector.
    LPIPS is omitted if unavailable and ``lpips_available`` is explicitly 0.
    Nonfinite RGB/mask/structure inputs raise before metric computation.
    """
    _validate_image_mask(gt, M)
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have the same shape.")
    _validate_metric_options(edge_scale, edge_threshold, edge_tolerance, orientation_threshold, boundary_width)
    pred, gt, M = pred.float(), gt.float(), M.float()
    _require_finite(pred=pred, gt=gt, M=M)
    results: List[Dict[str, float]] = []
    completed = composite_completed(pred, gt, M)
    gradient_pred, gradient_gt = grad(completed), grad(gt)
    pred_edges = (edge_strength(gradient_pred, scale=edge_scale) > edge_threshold).float()
    gt_edges = (edge_strength(gradient_gt, scale=edge_scale) > edge_threshold).float()
    similarity = ssim_map(completed, gt)
    batch_size = pred.shape[0]
    for idx in range(batch_size):
        gt_i = gt[idx : idx + 1]
        known_mask_i = M[idx : idx + 1]
        completed_i = completed[idx : idx + 1]
        gradient_i, target_i = gradient_pred[idx:idx + 1], gradient_gt[idx:idx + 1]
        gradient_diff = (gradient_i - target_i).abs()
        absolute_error = (completed_i - gt_i).abs()
        squared_error = absolute_error.square()
        ssim_i = similarity[idx:idx + 1]
        pred_edge_i, gt_edge_i = pred_edges[idx:idx + 1], gt_edges[idx:idx + 1]
        regions = _hole_regions(known_mask_i, boundary_width)
        values = {
            "hole_ratio": regions["hole"].mean(),
            "psnr": psnr(completed_i, gt_i),
            "ssim": ssim_i.mean(),
            "l1": absolute_error.mean(),
            "gradient_l1": gradient_diff.mean(),
            "boundary_consistency": boundary_consistency(completed_i, gt_i, known_mask_i),
        }
        for prefix, region in regions.items():
            mse = _region_mean(squared_error, region)
            values.update({
                f"{prefix}_pixels": region.sum(),
                f"{prefix}_psnr": 10 * torch.log10(4.0 / mse.clamp_min(1e-8)),
                f"{prefix}_ssim": _region_mean(ssim_i, region, empty_value=1),
                f"{prefix}_l1": _region_mean(absolute_error, region),
                f"{prefix}_gradient_l1": _region_mean(gradient_diff, region),
            })
        for prefix, region in (("edge", torch.ones_like(known_mask_i)), ("hole_edge", regions["hole"])):
            values.update({f"{prefix}_{name}": value for name, value in
                           _edge_scores(pred_edge_i, gt_edge_i, region, edge_tolerance).items()})
        values.update({f"hole_orientation_{name}": value for name, value in
                       _orientation_metrics(gradient_i, target_i, regions["hole"], orientation_threshold).items()})
        item = {name: float(value) for name, value in values.items()}
        item["index"] = float(idx)
        item["lpips_available"] = 0.0
        if lpips_metric is not None:
            lpips_whole = lpips_metric(completed_i, gt_i)
            if lpips_whole is not None and math.isfinite(float(lpips_whole)):
                item["lpips"] = float(lpips_whole)
                item["lpips_available"] = 1.0
        results.append(item)
    return results


@torch.no_grad()
def evaluate_structure_per_image(
    structure: dict | None,
    gt: torch.Tensor,
    M: torch.Tensor,
    edge_scale: float = 0.1,
    orientation_threshold: float = 0.05,
    *,
    edge_threshold: float = 0.5,
    edge_tolerance: int = 1,
) -> List[Dict[str, float]]:
    """Score full-resolution predicted G/E in holes against GT-derived targets.

    Returns one numeric dictionary per image, suitable for merging with
    ``evaluate_per_image``. ``structure=None`` returns empty dictionaries
    for each image, keeping batch indexing stable without inventing scores.
    G is [B,6,H,W]; E is [B,1,H,W] (or sigmoid of ``edge_logits``).
    ``structure_gradient_edge_*`` scores the edges implied by G, while
    ``structure_edge_*`` scores E. Angle/support conventions match RGB metrics.
    """
    _validate_image_mask(gt, M)
    _validate_metric_options(edge_scale, edge_threshold, edge_tolerance, orientation_threshold)
    if structure is None:
        return [{} for _ in range(gt.shape[0])]
    gradient = structure["gradient"].float()
    edge_name = "edge" if "edge" in structure else "edge_logits"
    edge_input = structure[edge_name].float()
    _require_finite(structure_gradient=gradient, **{f"structure_{edge_name}": edge_input})
    # Check logits before sigmoid: infinite logits would otherwise appear finite.
    if "edge_logits" in structure and edge_name != "edge_logits":
        _require_finite(structure_edge_logits=structure["edge_logits"])
    edge = edge_input if edge_name == "edge" else edge_input.sigmoid()
    if gradient.shape != (gt.shape[0], 6, gt.shape[2], gt.shape[3]) or edge.shape != M.shape:
        raise ValueError("Structure G/E must match the full-resolution image with 6/1 channels.")
    gt, M = gt.float(), M.float()
    _require_finite(gt=gt, M=M)
    target = grad(gt)
    target_edge = edge_strength(target, scale=edge_scale)
    implied_edge = edge_strength(gradient, scale=edge_scale)
    result = []
    for idx in range(gt.shape[0]):
        g, t = gradient[idx:idx + 1], target[idx:idx + 1]
        e, te = edge[idx:idx + 1], target_edge[idx:idx + 1]
        ge = implied_edge[idx:idx + 1]
        hole = 1 - M[idx:idx + 1]
        values = {
            "structure_hole_pixels": hole.sum(),
            "structure_gradient_l1": _region_mean((g - t).abs(), hole),
            "structure_edge_mae": _region_mean((e - te).abs(), hole),
            "structure_edge_brier": _region_mean((e - te).square(), hole),
            "structure_consistency_l1": _region_mean((e - ge).abs(), hole),
        }
        values.update({f"structure_orientation_{name}": value for name, value in
                       _orientation_metrics(g, t, hole, orientation_threshold).items()})
        for prefix, prediction in (("structure_edge", e), ("structure_gradient_edge", ge)):
            values.update({f"{prefix}_{name}": value for name, value in
                           _edge_scores(prediction > edge_threshold, te > edge_threshold, hole, edge_tolerance).items()})
        result.append({name: float(value) for name, value in values.items()})
    return result


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


def metric_sample_supported(item: Dict[str, float], name: str) -> bool:
    """Whether this per-image value has support for an image-averaged metric.

    Pixel/edge counts and availability flags retain zeros in their summaries.
    Region metrics require nonempty corresponding regions; orientation error
    also requires strong GT vectors. An empty edge set within a nonempty hole
    remains a valid edge score. Older rows lacking support counters retain
    their previous eligibility, while current evaluators always emit counters.
    """
    if name not in item or not math.isfinite(float(item[name])):
        return False
    if name.endswith(("_pixels", "_count", "_available")):
        return True
    support_keys = []
    if name.endswith("_orientation_error"):
        support_keys.append(name.removesuffix("_error") + "_count")
    for prefix, support in (
        ("hole_inner_", "hole_inner_pixels"),
        ("hole_boundary_", "hole_boundary_pixels"),
        ("hole_", "hole_pixels"),
        ("structure_", "structure_hole_pixels"),
        ("effective_prior_", "effective_prior_hole_pixels"),
    ):
        if name.startswith(prefix):
            support_keys.append(support)
            break
    return all(key not in item or (math.isfinite(float(item[key])) and float(item[key]) > 0)
               for key in support_keys)


def summarize_metric_items(items: List[Dict[str, float]]) -> Dict[str, float]:
    """Average supported image values; omit metrics with no supported image."""
    summary: Dict[str, float] = {}
    for name in metric_names_from_items(items):
        values = [float(item[name]) for item in items if metric_sample_supported(item, name)]
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
            values = [float(item[name]) for item in bucket_items if metric_sample_supported(item, name)]
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
