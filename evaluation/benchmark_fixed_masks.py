from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.image import pil_to_tensor, tensor_to_pil
from fg_elastica_inpaint.utils.metrics import (
    FrechetInceptionDistance,
    OptionalLPIPS,
    composite_completed,
    evaluate_per_image,
)


DEFAULT_BUCKETS = [
    ("10-20%", 0.10, 0.20),
    ("20-30%", 0.20, 0.30),
    ("30-40%", 0.30, 0.40),
    ("40-50%", 0.40, 0.50),
]
SUMMARY_BUCKET = "Mixed"
METRIC_COLUMNS = ["psnr", "ssim", "lpips", "fid"]
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_file_list(path: str | Path, limit: int | None = None) -> List[Path]:
    list_path = Path(path)
    if not list_path.is_file():
        raise FileNotFoundError(f"File list not found: {list_path}")
    paths = [Path(line.strip()) for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        paths = paths[:limit]
    return paths


def parse_bucket_spec(spec: str) -> tuple[str, float, float]:
    try:
        low_text, high_text = spec.rstrip("%").split("-", maxsplit=1)
        low = float(low_text) / 100.0
        high = float(high_text) / 100.0
    except Exception as exc:
        raise ValueError(f"Bucket must look like '10-20' or '10-20%', got {spec!r}") from exc
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"Invalid bucket range: {spec!r}")
    return f"{int(low * 100)}-{int(high * 100)}%", low, high


def bucket_for_ratio(hole_ratio: float, buckets: Sequence[tuple[str, float, float]]) -> str | None:
    for name, low, high in buckets:
        if low <= hole_ratio < high:
            return name
    return None


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "item"


def load_rgb(path: Path, image_size: int, resize_short_to: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    short_side = min(width, height)
    if short_side != resize_short_to:
        if width < height:
            new_width = resize_short_to
            new_height = int(round(height * resize_short_to / width))
        else:
            new_height = resize_short_to
            new_width = int(round(width * resize_short_to / height))
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    width, height = image.size
    left = max((width - image_size) // 2, 0)
    top = max((height - image_size) // 2, 0)
    image = image.crop((left, top, left + image_size, top + image_size))
    return pil_to_tensor(image, value_range="minus_one_to_one").unsqueeze(0)


def load_known_mask(path: Path, image_size: int, invert_mask: bool) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)
    if invert_mask:
        mask_np = 1.0 - mask_np
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)


def case_pairs(image_paths: Sequence[Path], mask_paths: Sequence[Path], pairing: str) -> Iterable[tuple[int, Path, Path]]:
    if pairing == "pairwise":
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Pairwise mode requires equal list lengths, got {len(image_paths)} images and {len(mask_paths)} masks."
            )
        for index, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            yield index, image_path, mask_path
        return

    case_index = 0
    for image_path in image_paths:
        for mask_path in mask_paths:
            yield case_index, image_path, mask_path
            case_index += 1


def metric_summary(items: Sequence[Dict[str, float]], bucket_name: str) -> Dict[str, float | str | int]:
    row: Dict[str, float | str | int] = {
        "bucket": bucket_name,
        "num_samples": len(items),
        "avg_mask_ratio": float("nan"),
    }
    if items:
        row["avg_mask_ratio"] = sum(float(item["hole_ratio"]) for item in items) / len(items)
    for name in METRIC_COLUMNS:
        values = [float(item[name]) for item in items if name in item and math.isfinite(float(item[name]))]
        row[name] = sum(values) / len(values) if values else float("nan")
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        return
    if fieldnames is None:
        names = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    else:
        names = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_number(value: object, precision: int) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.{precision}f}"


def write_latex_table(path: Path, rows: Sequence[Dict[str, object]], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Mask Ratio & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ & FID $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['bucket']} & "
            f"{latex_number(row.get('psnr'), 2)} & "
            f"{latex_number(row.get('ssim'), 4)} & "
            f"{latex_number(row.get('lpips'), 4)} & "
            f"{latex_number(row.get('fid'), 2)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def label_image(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 24), "white")
    canvas.paste(image.convert("RGB"), (0, 24))
    ImageDraw.Draw(canvas).text((6, 4), label, fill="black", font=font)
    return canvas


def save_comparison(gt: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, comp: torch.Tensor, path: Path) -> None:
    mask_np = (mask[0, 0].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    mask_image = Image.fromarray(mask_np, mode="L").convert("RGB")
    panels = [
        ("GT", tensor_to_pil(gt)),
        ("Masked", tensor_to_pil(gt * mask)),
        ("Mask", mask_image),
        ("Pred", tensor_to_pil(pred)),
        ("Composite", tensor_to_pil(comp)),
    ]
    font = ImageFont.load_default()
    labelled = [label_image(image, title, font) for title, image in panels]
    sheet = Image.new("RGB", (sum(image.width for image in labelled), labelled[0].height), "white")
    offset = 0
    for image in labelled:
        sheet.paste(image, (offset, 0))
        offset += image.width
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_case_images(
    output_root: Path,
    bucket_names: Sequence[str],
    case_id: str,
    gt: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    comp: torch.Tensor,
    mask_source: Path,
    save_pred: bool,
) -> None:
    for bucket_name in bucket_names:
        case_dir = output_root / "images" / safe_name(bucket_name) / safe_name(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(gt).save(case_dir / "gt.png")
        tensor_to_pil(gt * mask).save(case_dir / "masked.png")
        tensor_to_pil(comp).save(case_dir / "composite.png")
        if save_pred:
            tensor_to_pil(pred).save(case_dir / "pred.png")
        mask_np = (mask[0, 0].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(mask_np, mode="L").save(case_dir / "mask.png")
        save_comparison(gt, mask, pred, comp, case_dir / "comparison.png")
        (case_dir / "source.txt").write_text(f"mask={mask_source}\n", encoding="utf-8")


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a checkpoint on a fixed test list and fixed masks.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_list", type=str, required=True)
    parser.add_argument("--mask_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pairing", choices=["pairwise", "cross"], default="pairwise")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--buckets", nargs="*", default=[name for name, _, _ in DEFAULT_BUCKETS])
    parser.add_argument("--metric_target", choices=["composite", "pred"], default="composite")
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--compute_fid", action="store_true")
    parser.add_argument("--invert_mask", action="store_true", help="Use when masks are white=hole and black=known.")
    parser.add_argument("--save_pred", action="store_true", help="Also save raw pred.png in each case directory.")
    parser.add_argument("--no_save_images", action="store_true")
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--caption", default="Fixed-mask inpainting benchmark.")
    parser.add_argument("--label", default="tab:fixed_mask_benchmark")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 256))
    resize_short_to = int(data_cfg.get("resize_short_to", image_size))
    buckets = [parse_bucket_spec(spec) for spec in args.buckets]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(max(1, args.num_threads))
    elif device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    image_paths = read_file_list(args.test_list, limit=args.limit)
    mask_paths = read_file_list(args.mask_list, limit=args.limit if args.pairing == "pairwise" else None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = FeatureGuidedElasticaADMMNet.from_config(cfg).to(device)
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    lpips_metric = OptionalLPIPS(device) if args.compute_lpips else None
    fid_metrics = (
        {name: FrechetInceptionDistance(device) for name, _, _ in buckets} | {SUMMARY_BUCKET: FrechetInceptionDistance(device)}
        if args.compute_fid
        else {}
    )

    per_image_rows: List[Dict[str, object]] = []
    metric_items_by_bucket: Dict[str, List[Dict[str, float]]] = {name: [] for name, _, _ in buckets}
    metric_items_by_bucket[SUMMARY_BUCKET] = []

    total_cases = len(image_paths) * len(mask_paths) if args.pairing == "cross" else len(image_paths)
    iterator = tqdm(case_pairs(image_paths, mask_paths, args.pairing), total=total_cases, desc="benchmark")
    for case_index, image_path, mask_path in iterator:
        if image_path.suffix.lower() not in VALID_IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {image_path}")
        gt = load_rgb(image_path, image_size, resize_short_to).to(device)
        mask = load_known_mask(mask_path, image_size, args.invert_mask).to(device)
        masked = gt * mask

        outputs = model(masked, mask)
        pred = outputs["pred"]
        comp = composite_completed(pred, gt, mask)
        metric_tensor = comp if args.metric_target == "composite" else pred
        item = evaluate_per_image(metric_tensor, gt, mask, lpips_metric)[0]
        hole_ratio = float(item["hole_ratio"])
        bucket_name = bucket_for_ratio(hole_ratio, buckets)

        case_id = f"{case_index:06d}_{safe_name(image_path.stem)}_{safe_name(mask_path.stem)}"
        row: Dict[str, object] = {
            "case_id": case_id,
            "image": str(image_path),
            "mask": str(mask_path),
            "bucket": bucket_name or "outside",
            "metric_target": args.metric_target,
            **{key: value for key, value in item.items() if key != "index"},
        }
        per_image_rows.append(row)

        metric_items_by_bucket[SUMMARY_BUCKET].append(item)
        if bucket_name is not None:
            metric_items_by_bucket[bucket_name].append(item)

        if fid_metrics:
            generated = metric_tensor
            fid_metrics[SUMMARY_BUCKET].update(generated, gt)
            if bucket_name is not None:
                fid_metrics[bucket_name].update(generated, gt)

        if not args.no_save_images:
            image_buckets = [SUMMARY_BUCKET]
            if bucket_name is not None:
                image_buckets.insert(0, bucket_name)
            save_case_images(output_dir, image_buckets, case_id, gt.cpu(), mask.cpu(), pred.cpu(), comp.cpu(), mask_path, args.save_pred)

    summary_rows = []
    for bucket_name, _, _ in buckets:
        summary_rows.append(metric_summary(metric_items_by_bucket[bucket_name], bucket_name))
    summary_rows.append(metric_summary(metric_items_by_bucket[SUMMARY_BUCKET], SUMMARY_BUCKET))

    if fid_metrics:
        for row in summary_rows:
            bucket_name = str(row["bucket"])
            if int(row["num_samples"]) <= 0:
                row["fid"] = float("nan")
                continue
            try:
                row["fid"] = fid_metrics[bucket_name].compute()
            except Exception as exc:
                row["fid"] = float("nan")
                row["fid_error"] = str(exc)

    per_image_fields = [
        "case_id",
        "image",
        "mask",
        "bucket",
        "metric_target",
        "hole_ratio",
        "psnr",
        "ssim",
        "l1",
        "edge_f1",
        "gradient_l1",
        "boundary_consistency",
        "lpips",
    ]
    write_csv(output_dir / "per_image_metrics.csv", per_image_rows, per_image_fields)
    write_csv(output_dir / "summary_by_bucket.csv", summary_rows)
    write_latex_table(output_dir / "benchmark_table.tex", summary_rows, args.caption, args.label)
    with (output_dir / "benchmark_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": str(Path(args.config)),
                "checkpoint": str(Path(args.checkpoint)),
                "test_list": str(Path(args.test_list)),
                "mask_list": str(Path(args.mask_list)),
                "pairing": args.pairing,
                "metric_target": args.metric_target,
                "mask_convention": "white/1=known, black/0=hole" if not args.invert_mask else "input masks inverted; effective white/1=known",
                "summary": summary_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved benchmark outputs to {output_dir.resolve()}")
    print(f"Summary CSV: {(output_dir / 'summary_by_bucket.csv').resolve()}")
    print(f"LaTeX table: {(output_dir / 'benchmark_table.tex').resolve()}")


if __name__ == "__main__":
    main()
