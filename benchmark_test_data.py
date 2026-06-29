from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.image import pil_to_tensor, tensor_to_pil


HOLE_LEVELS = {
    "small": (0.10, 0.20),
    "medium": (0.20, 0.30),
    "large": (0.30, 0.45),
    "xlarge": (0.45, 0.60),
}


def load_rgb(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return pil_to_tensor(image, value_range="minus_one_to_one").unsqueeze(0)


def make_irregular_mask(size: int, target_ratio: float, seed: int) -> tuple[Image.Image, float]:
    """Create a deterministic, contiguous, irregular hole; white pixels are known."""
    rng = random.Random(seed)
    angles = [2.0 * math.pi * idx / 24.0 for idx in range(24)]
    radius_jitter = [rng.uniform(0.65, 1.30) for _ in angles]
    center_x = size * rng.uniform(0.43, 0.57)
    center_y = size * rng.uniform(0.43, 0.57)

    def render(scale: float) -> tuple[Image.Image, float]:
        mask = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(mask)
        points = [
            (
                center_x + scale * radius * math.cos(angle),
                center_y + scale * radius * math.sin(angle),
            )
            for angle, radius in zip(angles, radius_jitter)
        ]
        draw.polygon(points, fill=0)
        ratio = float((np.asarray(mask, dtype=np.uint8) == 0).mean())
        return mask, ratio

    low, high = 0.0, float(size)
    best_mask, best_ratio = render(0.0)
    for _ in range(20):
        scale = (low + high) * 0.5
        mask, ratio = render(scale)
        if abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
            best_mask, best_ratio = mask, ratio
        if ratio < target_ratio:
            low = scale
        else:
            high = scale
    return best_mask, best_ratio


def make_scattered_mask(size: int, target_ratio: float, seed: int) -> tuple[Image.Image, float]:
    """Create scattered holes with deliberately varied component sizes and shapes."""
    rng = random.Random(seed)
    grid_size = 5
    cells = [(row, col) for row in range(grid_size) for col in range(grid_size)]
    rng.shuffle(cells)
    component_count = 8 if target_ratio <= 0.20 else 13 if target_ratio <= 0.30 else 19 if target_ratio <= 0.45 else 24
    components = []
    cell_size = size / grid_size
    for row, col in cells[:component_count]:
        center_x = (col + rng.uniform(0.16, 0.84)) * cell_size
        center_y = (row + rng.uniform(0.16, 0.84)) * cell_size
        size_factor = rng.uniform(0.28, 0.52) if rng.random() < 0.72 else rng.uniform(0.68, 1.15)
        aspect = rng.uniform(0.38, 2.4)
        radius_x = cell_size * size_factor * math.sqrt(aspect)
        radius_y = cell_size * size_factor / math.sqrt(aspect)
        vertex_count = rng.randint(5, 14)
        angles = [2.0 * math.pi * idx / vertex_count for idx in range(vertex_count)]
        jitter = [rng.uniform(0.48, 1.40) for _ in angles]
        component_type = "stroke" if rng.random() < 0.25 else "blob"
        angle = rng.uniform(0.0, 2.0 * math.pi)
        components.append((component_type, center_x, center_y, radius_x, radius_y, angles, jitter, angle))

    def render(scale: float) -> tuple[Image.Image, float]:
        mask = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(mask)
        for component_type, center_x, center_y, radius_x, radius_y, angles, jitter, angle in components:
            if component_type == "stroke":
                length = scale * max(radius_x, radius_y) * 2.5
                width = max(2, int(scale * min(radius_x, radius_y) * 0.75))
                dx, dy = math.cos(angle) * length * 0.5, math.sin(angle) * length * 0.5
                draw.line([(center_x - dx, center_y - dy), (center_x + dx, center_y + dy)], fill=0, width=width)
                cap = width * 0.5
                for x, y in ((center_x - dx, center_y - dy), (center_x + dx, center_y + dy)):
                    draw.ellipse([x - cap, y - cap, x + cap, y + cap], fill=0)
            else:
                points = [
                    (
                        center_x + scale * radius_x * factor * math.cos(theta),
                        center_y + scale * radius_y * factor * math.sin(theta),
                    )
                    for theta, factor in zip(angles, jitter)
                ]
                draw.polygon(points, fill=0)
        ratio = float((np.asarray(mask, dtype=np.uint8) == 0).mean())
        return mask, ratio

    low, high = 0.0, 3.0
    best_mask, best_ratio = render(0.0)
    for _ in range(20):
        scale = (low + high) * 0.5
        mask, ratio = render(scale)
        if abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
            best_mask, best_ratio = mask, ratio
        if ratio < target_ratio:
            low = scale
        else:
            high = scale
    return best_mask, best_ratio


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    values = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0)


def psnr(mse: torch.Tensor) -> float:
    return float((-10.0 * torch.log10(mse.clamp_min(1e-10))).item())


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() * values.shape[1]).clamp_min(1.0)


def gaussian_window(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(11, device=device, dtype=dtype) - 5.0
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, 11, 11).contiguous()


def ssim_global(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = ((pred.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
    gt = ((gt.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
    window = gaussian_window(pred.shape[1], pred.device, pred.dtype)
    mu_pred = F.conv2d(pred, window, padding=5, groups=pred.shape[1])
    mu_gt = F.conv2d(gt, window, padding=5, groups=gt.shape[1])
    sigma_pred = F.conv2d(pred * pred, window, padding=5, groups=pred.shape[1]) - mu_pred.square()
    sigma_gt = F.conv2d(gt * gt, window, padding=5, groups=gt.shape[1]) - mu_gt.square()
    sigma_cross = F.conv2d(pred * gt, window, padding=5, groups=pred.shape[1]) - mu_pred * mu_gt
    c1, c2 = 0.01**2, 0.03**2
    score = ((2.0 * mu_pred * mu_gt + c1) * (2.0 * sigma_cross + c2))
    score = score / ((mu_pred.square() + mu_gt.square() + c1) * (sigma_pred + sigma_gt + c2))
    return float(score.mean().item())


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, known_mask: torch.Tensor) -> dict[str, float]:
    pred_01 = ((pred.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
    gt_01 = ((gt.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
    error = pred_01 - gt_01
    squared_error = error.square()
    absolute_error = error.abs()
    hole_mask = 1.0 - known_mask
    mse_global = squared_error.mean()
    mse_hole = masked_mean(squared_error, hole_mask)
    return {
        "mse_global": float(mse_global.item()),
        "mae_global": float(absolute_error.mean().item()),
        "psnr_global": psnr(mse_global),
        "ssim_global": ssim_global(pred, gt),
        "mse_hole": float(mse_hole.item()),
        "mae_hole": float(masked_mean(absolute_error, hole_mask).item()),
        "psnr_hole": psnr(mse_hole),
    }


def save_masked_image(image: torch.Tensor, known_mask: torch.Tensor, path: Path) -> None:
    tensor_to_pil(image * known_mask).save(path)


def label_image(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 24), "white")
    canvas.paste(image.convert("RGB"), (0, 24))
    ImageDraw.Draw(canvas).text((6, 4), label, fill="black", font=font)
    return canvas


def save_comparison(gt: torch.Tensor, known_mask: torch.Tensor, pred: torch.Tensor, comp: torch.Tensor, path: Path) -> None:
    mask_image = Image.fromarray((known_mask[0, 0].detach().cpu().numpy() * 255.0).round().astype(np.uint8), mode="L")
    panels = [
        ("GT", tensor_to_pil(gt)),
        ("Masked", tensor_to_pil(gt * known_mask)),
        ("Mask", mask_image.convert("RGB")),
        ("Pred", tensor_to_pil(pred)),
        ("Composite", tensor_to_pil(comp)),
    ]
    font = ImageFont.load_default()
    labelled = [label_image(image, label, font) for label, image in panels]
    sheet = Image.new("RGB", (sum(image.width for image in labelled), labelled[0].height), "white")
    offset = 0
    for image in labelled:
        sheet.paste(image, (offset, 0))
        offset += image.width
    sheet.save(path)


def save_mask_preview(gt: torch.Tensor, known_mask: torch.Tensor, path: Path) -> None:
    mask_image = Image.fromarray((known_mask[0, 0].detach().cpu().numpy() * 255.0).round().astype(np.uint8), mode="L")
    font = ImageFont.load_default()
    panels = [
        label_image(tensor_to_pil(gt), "GT", font),
        label_image(tensor_to_pil(gt * known_mask), "Masked", font),
        label_image(mask_image.convert("RGB"), "Mask", font),
    ]
    sheet = Image.new("RGB", (sum(image.width for image in panels), panels[0].height), "white")
    offset = 0
    for image in panels:
        sheet.paste(image, (offset, 0))
        offset += image.width
    sheet.save(path)


def save_contact_sheet(rows: list[tuple[str, Path]], path: Path) -> None:
    images = [(label, Image.open(image_path).convert("RGB")) for label, image_path in rows]
    font = ImageFont.load_default()
    width = max(image.width for _, image in images)
    height = sum(image.height + 22 for _, image in images)
    sheet = Image.new("RGB", (width, height), "white")
    offset = 0
    draw = ImageDraw.Draw(sheet)
    for label, image in images:
        draw.text((6, offset + 3), label, fill="black", font=font)
        sheet.paste(image, (0, offset + 22))
        offset += image.height + 22
    sheet.save(path)


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["hole_level"]), str(row["output_type"]))].append(row)
    summary = []
    metrics = [key for key in rows[0] if key.endswith(("_global", "_hole"))]
    for (hole_level, output_type), values in grouped.items():
        item: dict[str, float | str] = {
            "hole_level": hole_level,
            "output_type": output_type,
            "samples": len(values),
            "mean_hole_ratio": mean(float(value["hole_ratio"]) for value in values),
        }
        for metric in metrics:
            numbers = [float(value[metric]) for value in values]
            item[f"{metric}_mean"] = mean(numbers)
            item[f"{metric}_std"] = stdev(numbers) if len(numbers) > 1 else 0.0
        summary.append(item)
    return summary


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="CPU benchmark for fixed test images and synthetic hole sizes.")
    parser.add_argument("--config", default="output/config_resolved.yaml")
    parser.add_argument("--checkpoint", default="output/best.pt")
    parser.add_argument("--input_dir", default="test_data")
    parser.add_argument("--output_dir", default="output/cpu_mask_benchmark")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--resume", action="store_true", help="Reuse completed cases in output_dir.")
    parser.add_argument("--mask_topology", choices=("contiguous", "scattered"), default="contiguous")
    parser.add_argument("--masks_only", action="store_true", help="Generate preview masks without model inference.")
    parser.add_argument("--mask_dir", help="Read fixed masks from this directory instead of generating them.")
    parser.add_argument("--image_name", help="Run only one image filename from input_dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    image_size = int(config["data"].get("image_size", 256))
    input_dir = Path(args.input_dir)
    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.image_name:
        image_paths = [path for path in image_paths if path.name == args.image_name]
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {input_dir.resolve()}")
    mask_dir = Path(args.mask_dir) if args.mask_dir else None

    device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = None
    if not args.masks_only:
        model = FeatureGuidedElasticaADMMNet.from_config(config).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    contact_rows: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for image_index, image_path in enumerate(image_paths):
        gt = load_rgb(image_path, image_size).to(device)
        for level_index, (level_name, (low, high)) in enumerate(HOLE_LEVELS.items()):
            target_ratio = (low + high) * 0.5
            for variant in range(args.variants):
                seed = args.seed + image_index * 10_000 + level_index * 100 + variant
                case_dir = output_dir / image_path.stem / f"{level_name}_v{variant + 1}"
                source_case_dir = mask_dir / image_path.stem / f"{level_name}_v{variant + 1}" if mask_dir else None
                completed = all(
                    (case_dir / filename).is_file()
                    for filename in ("mask.png", "pred.png", "composite.png", "comparison.png")
                )
                if args.resume and completed and not args.masks_only:
                    mask_image = Image.open(case_dir / "mask.png").convert("L")
                    pred = load_rgb(case_dir / "pred.png", image_size).to(device)
                    comp = load_rgb(case_dir / "composite.png", image_size).to(device)
                elif source_case_dir is not None:
                    mask_path = source_case_dir / "mask.png"
                    if not mask_path.is_file():
                        raise FileNotFoundError(f"Fixed mask not found: {mask_path.resolve()}")
                    mask_image = Image.open(mask_path).convert("L")
                else:
                    mask_builder = make_scattered_mask if args.mask_topology == "scattered" else make_irregular_mask
                    mask_image, _ = mask_builder(image_size, target_ratio, seed)
                known_mask = mask_to_tensor(mask_image).to(device)
                hole_ratio = float((1.0 - known_mask).mean().item())
                if args.masks_only:
                    case_dir.mkdir(parents=True, exist_ok=True)
                    tensor_to_pil(gt).save(case_dir / "gt.png")
                    mask_image.save(case_dir / "mask.png")
                    save_masked_image(gt, known_mask, case_dir / "masked.png")
                    save_mask_preview(gt, known_mask, case_dir / "mask_preview.png")
                    contact_rows[image_path.stem].append(
                        (f"{level_name} | hole={hole_ratio:.3f} | variant={variant + 1}", case_dir / "mask_preview.png")
                    )
                    print(f"Generated {image_path.name}: {level_name}, variant {variant + 1}, hole={hole_ratio:.3f}", flush=True)
                    continue
                if not (args.resume and completed):
                    masked = gt * known_mask
                    outputs = model(masked, known_mask)
                    pred, comp = outputs["pred"], outputs["comp"]
                    case_dir.mkdir(parents=True, exist_ok=True)
                    tensor_to_pil(gt).save(case_dir / "gt.png")
                    mask_image.save(case_dir / "mask.png")
                    save_masked_image(gt, known_mask, case_dir / "masked.png")
                    tensor_to_pil(pred).save(case_dir / "pred.png")
                    tensor_to_pil(comp).save(case_dir / "composite.png")
                    save_comparison(gt, known_mask, pred, comp, case_dir / "comparison.png")
                contact_rows[image_path.stem].append(
                    (f"{level_name} | hole={hole_ratio:.3f} | variant={variant + 1}", case_dir / "comparison.png")
                )

                for output_type, output in (("pred", pred), ("composite", comp)):
                    row: dict[str, float | str] = {
                        "image": image_path.name,
                        "hole_level": level_name,
                        "variant": variant + 1,
                        "seed": seed,
                        "hole_ratio": hole_ratio,
                        "output_type": output_type,
                    }
                    row.update(compute_metrics(output, gt, known_mask))
                    rows.append(row)
                status = "Reused" if args.resume and completed else "Completed"
                print(f"{status} {image_path.name}: {level_name}, variant {variant + 1}, hole={hole_ratio:.3f}", flush=True)

    if rows:
        write_csv(output_dir / "metrics.csv", rows)
        write_csv(output_dir / "metrics_summary.csv", summarize(rows))
    for image_name, sheet_rows in contact_rows.items():
        save_contact_sheet(sheet_rows, output_dir / f"contact_sheet_{image_name}.png")
    print(f"Saved benchmark outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
