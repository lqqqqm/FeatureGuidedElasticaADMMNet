from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.image import pil_to_tensor, tensor_to_pil


def load_rgb(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return pil_to_tensor(image, value_range="minus_one_to_one").unsqueeze(0)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    values = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    values = (values > 0.5).astype(np.float32)
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0)


def save_masked_image(gt: torch.Tensor, known_mask: torch.Tensor, path: Path) -> None:
    tensor_to_pil(gt * known_mask).save(path)


def label_image(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 24), "white")
    canvas.paste(image.convert("RGB"), (0, 24))
    ImageDraw.Draw(canvas).text((6, 4), label, fill="black", font=font)
    return canvas


def save_comparison(gt: torch.Tensor, known_mask: torch.Tensor, pred: torch.Tensor, comp: torch.Tensor, path: Path) -> None:
    mask_image = Image.fromarray((known_mask[0, 0].detach().cpu().numpy() * 255.0).round().astype(np.uint8), mode="L")
    font = ImageFont.load_default()
    panels = [
        ("GT", tensor_to_pil(gt)),
        ("Masked", tensor_to_pil(gt * known_mask)),
        ("Mask", mask_image.convert("RGB")),
        ("Pred", tensor_to_pil(pred)),
        ("Composite", tensor_to_pil(comp)),
    ]
    labelled = [label_image(image, label, font) for label, image in panels]
    sheet = Image.new("RGB", (sum(image.width for image in labelled), labelled[0].height), "white")
    x = 0
    for image in labelled:
        sheet.paste(image, (x, 0))
        x += image.width
    sheet.save(path)


def save_contact_sheet(rows: list[tuple[str, Path]], path: Path) -> None:
    images = [(label, Image.open(image_path).convert("RGB")) for label, image_path in rows]
    font = ImageFont.load_default()
    width = max(image.width for _, image in images)
    height = sum(image.height + 22 for _, image in images)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for label, image in images:
        draw.text((6, y + 3), label, fill="black", font=font)
        sheet.paste(image, (0, y + 22))
        y += image.height + 22
    sheet.save(path)


def make_center_square(size: int, side: int) -> Image.Image:
    mask = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(mask)
    x1 = (size - side) // 2
    y1 = (size - side) // 2
    draw.rectangle([x1, y1, x1 + side - 1, y1 + side - 1], fill=0)
    return mask


def make_center_disk(size: int, target_ratio: float) -> Image.Image:
    mask = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(mask)
    radius = math.sqrt(target_ratio * size * size / math.pi)
    cx = cy = size / 2.0
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=0)
    return mask


def make_band(size: int, width: int, vertical: bool) -> Image.Image:
    mask = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(mask)
    if vertical:
        x1 = (size - width) // 2
        draw.rectangle([x1, 0, x1 + width - 1, size - 1], fill=0)
    else:
        y1 = (size - width) // 2
        draw.rectangle([0, y1, size - 1, y1 + width - 1], fill=0)
    return mask


def make_grid_squares(size: int, count_per_side: int, square_side: int) -> Image.Image:
    mask = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(mask)
    if count_per_side == 1:
        return make_center_square(size, square_side)
    gap = (size - count_per_side * square_side) / (count_per_side + 1)
    for row in range(count_per_side):
        for col in range(count_per_side):
            x1 = round(gap + col * (square_side + gap))
            y1 = round(gap + row * (square_side + gap))
            draw.rectangle([x1, y1, x1 + square_side - 1, y1 + square_side - 1], fill=0)
    return mask


def build_cases(size: int) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for side in (64, 96, 128, 160):
        cases.append(
            {
                "case_id": f"center_square_{side}",
                "group": "center_square_ramp",
                "description": f"center square side={side}",
                "mask": make_center_square(size, side),
            }
        )
    for count, side in ((1, 128), (2, 64), (4, 32)):
        cases.append(
            {
                "case_id": f"same_area_{count * count}_components",
                "group": "same_area_connectivity",
                "description": f"same area, {count * count} square component(s)",
                "mask": make_grid_squares(size, count, side),
            }
        )
    for case_id, description, mask in (
        ("same_area_horizontal_band", "same area, horizontal band", make_band(size, 64, vertical=False)),
        ("same_area_vertical_band", "same area, vertical band", make_band(size, 64, vertical=True)),
        ("same_area_center_square", "same area, center square", make_center_square(size, 128)),
        ("same_area_center_disk", "same area, center disk", make_center_disk(size, 0.25)),
    ):
        cases.append(
            {
                "case_id": case_id,
                "group": "same_area_distance",
                "description": description,
                "mask": mask,
            }
        )
    return cases


def distance_stats(mask: Image.Image) -> dict[str, float]:
    known = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
    hole = ~known
    distances = distance_transform_edt(hole)
    hole_distances = distances[hole]
    return {
        "hole_ratio": float(hole.mean()),
        "max_distance_to_known": float(hole_distances.max(initial=0.0)),
        "mean_distance_to_known": float(hole_distances.mean() if hole_distances.size else 0.0),
        "p90_distance_to_known": float(np.percentile(hole_distances, 90) if hole_distances.size else 0.0),
    }


def tensor_01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).float()


def psnr_from_mse(mse: float) -> float:
    return float(-10.0 * math.log10(max(mse, 1e-10)))


def ssim_global(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_01 = tensor_01(pred)
    gt_01 = tensor_01(gt)
    channels = pred_01.shape[1]
    coords = torch.arange(11, dtype=pred_01.dtype, device=pred_01.device) - 5.0
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).expand(channels, 1, 11, 11).contiguous()
    mu_x = F.conv2d(pred_01, kernel_2d, padding=5, groups=channels)
    mu_y = F.conv2d(gt_01, kernel_2d, padding=5, groups=channels)
    sigma_x = F.conv2d(pred_01 * pred_01, kernel_2d, padding=5, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(gt_01 * gt_01, kernel_2d, padding=5, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(pred_01 * gt_01, kernel_2d, padding=5, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2))
    score = score / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
    return float(score.mean().item())


def compute_metrics(pred: torch.Tensor, comp: torch.Tensor, gt: torch.Tensor, known_mask: torch.Tensor) -> dict[str, float]:
    pred_01 = tensor_01(pred)
    comp_01 = tensor_01(comp)
    gt_01 = tensor_01(gt)
    hole = 1.0 - known_mask
    hole3 = hole.repeat(1, 3, 1, 1)
    pred_err = (pred_01 - gt_01).abs()
    comp_err = (comp_01 - gt_01).abs()
    pred_sq = (pred_01 - gt_01).square()
    comp_sq = (comp_01 - gt_01).square()
    hole_den = hole3.sum().clamp_min(1.0)
    pred_mse_hole = float((pred_sq * hole3).sum().item() / float(hole_den.item()))
    comp_mse_global = float(comp_sq.mean().item())
    return {
        "pred_mae_hole": float((pred_err * hole3).sum().item() / float(hole_den.item())),
        "pred_mse_hole": pred_mse_hole,
        "pred_psnr_hole": psnr_from_mse(pred_mse_hole),
        "pred_mae_global": float(pred_err.mean().item()),
        "comp_mae_global": float(comp_err.mean().item()),
        "comp_mse_global": comp_mse_global,
        "comp_psnr_global": psnr_from_mse(comp_mse_global),
        "comp_ssim_global": ssim_global(comp, gt),
    }


def compute_distance_bins(pred: torch.Tensor, gt: torch.Tensor, mask: Image.Image, bin_width: int = 16) -> list[dict[str, float | str]]:
    pred_01 = tensor_01(pred)[0].detach().cpu().numpy()
    gt_01 = tensor_01(gt)[0].detach().cpu().numpy()
    abs_err = np.abs(pred_01 - gt_01).mean(axis=0)
    sq_err = ((pred_01 - gt_01) ** 2).mean(axis=0)
    known = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
    hole = ~known
    distances = distance_transform_edt(hole)
    rows: list[dict[str, float | str]] = []
    max_distance = int(math.ceil(float(distances[hole].max(initial=0.0))))
    for start in range(0, max_distance + 1, bin_width):
        end = start + bin_width
        region = hole & (distances > start) & (distances <= end)
        if not region.any():
            continue
        mse = float(sq_err[region].mean())
        rows.append(
            {
                "distance_bin": f"({start},{end}]",
                "distance_mid": float((start + end) / 2.0),
                "pixels": int(region.sum()),
                "mae": float(abs_err[region].mean()),
                "mse": mse,
                "psnr": psnr_from_mse(mse),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def aggregate_case_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    output = []
    for case_id, values in grouped.items():
        first = values[0]
        output.append(
            {
                "case_id": case_id,
                "group": first["group"],
                "description": first["description"],
                "hole_ratio": mean(float(row["hole_ratio"]) for row in values),
                "max_distance_to_known": mean(float(row["max_distance_to_known"]) for row in values),
                "mean_distance_to_known": mean(float(row["mean_distance_to_known"]) for row in values),
                "pred_mae_hole": mean(float(row["pred_mae_hole"]) for row in values),
                "pred_psnr_hole": mean(float(row["pred_psnr_hole"]) for row in values),
                "comp_psnr_global": mean(float(row["comp_psnr_global"]) for row in values),
                "comp_ssim_global": mean(float(row["comp_ssim_global"]) for row in values),
            }
        )
    return sorted(output, key=lambda row: (str(row["group"]), float(row["max_distance_to_known"])))


def aggregate_bin_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["distance_bin"])].append(row)
    output = []
    for distance_bin, values in grouped.items():
        output.append(
            {
                "distance_bin": distance_bin,
                "distance_mid": mean(float(row["distance_mid"]) for row in values),
                "samples": len(values),
                "mae": mean(float(row["mae"]) for row in values),
                "psnr": mean(float(row["psnr"]) for row in values),
            }
        )
    return sorted(output, key=lambda row: float(row["distance_mid"]))


def write_report(path: Path, case_summary: list[dict[str, object]], bin_summary: list[dict[str, object]], image_count: int) -> None:
    max_d = [float(row["max_distance_to_known"]) for row in case_summary]
    mean_d = [float(row["mean_distance_to_known"]) for row in case_summary]
    hole = [float(row["hole_ratio"]) for row in case_summary]
    mae = [float(row["pred_mae_hole"]) for row in case_summary]
    psnr_hole = [float(row["pred_psnr_hole"]) for row in case_summary]
    corr_max_mae = corr(max_d, mae)
    corr_mean_mae = corr(mean_d, mae)
    corr_hole_mae = corr(hole, mae)
    corr_max_psnr = corr(max_d, psnr_hole)

    lines = [
        "# Distance-to-known Mask Experiment",
        "",
        "## Setup",
        "",
        f"- Images: {image_count} image(s) from `test_data`.",
        "- Checkpoint: `output/best.pt`.",
        "- Mask convention: white/1 = known, black/0 = hole.",
        "- Main variable: distance from each hole pixel to the nearest known pixel.",
        "",
        "## Main Findings",
        "",
        f"- Correlation between max hole distance and hole MAE: `{corr_max_mae:.3f}`.",
        f"- Correlation between mean hole distance and hole MAE: `{corr_mean_mae:.3f}`.",
        f"- Correlation between hole ratio and hole MAE: `{corr_hole_mae:.3f}`.",
        f"- Correlation between max hole distance and hole PSNR: `{corr_max_psnr:.3f}`.",
        "",
        "A positive distance/MAE correlation and negative distance/PSNR correlation indicate that the method degrades as the unknown region gets farther from known context.",
        "",
        "## Case Summary",
        "",
        "| case | group | hole ratio | max dist | mean dist | hole MAE | hole PSNR | comp PSNR | comp SSIM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in case_summary:
        lines.append(
            "| {case_id} | {group} | {hole_ratio:.3f} | {max_distance_to_known:.1f} | "
            "{mean_distance_to_known:.1f} | {pred_mae_hole:.4f} | {pred_psnr_hole:.2f} | "
            "{comp_psnr_global:.2f} | {comp_ssim_global:.4f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Error by Distance Bin",
            "",
            "| distance bin | samples | MAE | PSNR |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in bin_summary:
        lines.append("| {distance_bin} | {samples} | {mae:.4f} | {psnr:.2f} |".format(**row))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `center_square_ramp` isolates larger continuous holes. It tests whether the center of a missing region collapses as the boundary gets farther away.",
            "- `same_area_connectivity` keeps the missing area near 25% but changes one large component into many small components. If fragmented masks score better, the bottleneck is long-range propagation rather than raw missing area.",
            "- `same_area_distance` keeps area near 25% but changes geometry. Bands have lower max distance than square/disk masks, so they test whether distance explains failures better than area.",
            "",
            "## Improvement Suggestions",
            "",
            "1. Add distance-aware training masks: oversample large contiguous square/disk holes and same-area geometry variants, not only random mixed masks.",
            "2. Add a distance-weighted reconstruction loss inside holes, e.g. give higher weight to pixels farther from the known boundary.",
            "3. Add multi-scale/global context before local elastica refinement, because far-from-boundary pixels need semantic priors more than boundary continuation.",
            "4. Evaluate with distance-bin metrics during validation, so improvements are visible specifically in the hard center pixels rather than hidden by global PSNR.",
            "5. Consider a two-stage strategy: coarse semantic fill for the whole hole, then edge/elastica-guided refinement near structures and boundaries.",
            "",
            "## Artifacts",
            "",
            "- `metrics.csv`: per image and per mask metrics.",
            "- `case_summary.csv`: metrics averaged across test images.",
            "- `distance_bin_summary.csv`: error averaged by distance-to-known bin.",
            "- `masks/`: generated masks.",
            "- `*/comparison.png`: visual comparisons for each image and mask.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="output/config_resolved.yaml")
    parser.add_argument("--checkpoint", default="output/best.pt")
    parser.add_argument("--input_dir", default="test_data")
    parser.add_argument("--output_dir", default="output/distance_mask_experiment")
    parser.add_argument("--image_name", help="Run only one image filename from input_dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    size = int(config["data"].get("image_size", 256))
    input_dir = Path(args.input_dir)
    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.image_name:
        image_paths = [path for path in image_paths if path.name == args.image_name]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir.resolve()}")

    device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model = FeatureGuidedElasticaADMMNet.from_config(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(size)
    case_manifest: list[dict[str, object]] = []
    for case in cases:
        mask = case["mask"]
        assert isinstance(mask, Image.Image)
        stats = distance_stats(mask)
        mask_path = masks_dir / f"{case['case_id']}.png"
        mask.save(mask_path)
        case_manifest.append({k: v for k, v in case.items() if k != "mask"} | stats | {"mask_path": str(mask_path)})
    write_csv(output_dir / "mask_manifest.csv", case_manifest)

    metric_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    contact_rows: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for image_path in image_paths:
        gt = load_rgb(image_path, size).to(device)
        for case, manifest_row in zip(cases, case_manifest):
            case_id = str(case["case_id"])
            mask = case["mask"]
            assert isinstance(mask, Image.Image)
            known_mask = mask_to_tensor(mask).to(device)
            masked = gt * known_mask
            outputs = model(masked, known_mask)
            pred = outputs["pred"]
            comp = outputs["comp"]

            case_dir = output_dir / image_path.stem / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            tensor_to_pil(gt).save(case_dir / "gt.png")
            mask.save(case_dir / "mask.png")
            save_masked_image(gt, known_mask, case_dir / "masked.png")
            tensor_to_pil(pred).save(case_dir / "pred.png")
            tensor_to_pil(comp).save(case_dir / "composite.png")
            save_comparison(gt, known_mask, pred, comp, case_dir / "comparison.png")

            row = {
                "image": image_path.name,
                "case_id": case_id,
                "group": case["group"],
                "description": case["description"],
                **{key: manifest_row[key] for key in ("hole_ratio", "max_distance_to_known", "mean_distance_to_known", "p90_distance_to_known")},
                **compute_metrics(pred, comp, gt, known_mask),
            }
            metric_rows.append(row)

            for bin_row in compute_distance_bins(pred, gt, mask):
                bin_rows.append({"image": image_path.name, "case_id": case_id, "group": case["group"], **bin_row})

            contact_rows[image_path.stem].append(
                (
                    f"{case_id} | hole={float(manifest_row['hole_ratio']):.3f} | maxD={float(manifest_row['max_distance_to_known']):.1f}",
                    case_dir / "comparison.png",
                )
            )
            print(f"Completed {image_path.name}: {case_id}", flush=True)

    case_summary = aggregate_case_rows(metric_rows)
    bin_summary = aggregate_bin_rows(bin_rows)
    write_csv(output_dir / "metrics.csv", metric_rows)
    write_csv(output_dir / "distance_bins.csv", bin_rows)
    write_csv(output_dir / "case_summary.csv", case_summary)
    write_csv(output_dir / "distance_bin_summary.csv", bin_summary)
    for image_name, rows in contact_rows.items():
        save_contact_sheet(rows, output_dir / f"contact_sheet_{image_name}.png")
    write_report(output_dir / "distance_mask_experiment_report.md", case_summary, bin_summary, len(image_paths))
    print(f"Saved experiment to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
