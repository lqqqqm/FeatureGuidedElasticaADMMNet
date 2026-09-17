"""Fixed-input, separate-process v0/V1 comparison. Never modifies model code.

Default paths target the user's GPU machine. See docs/compare_v0_v1.md.
This module deliberately imports no project package at module scope: the old
worker imports its helpers without importing the new model by accident.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD = Path("C:/codeee/inpainting/FeatureGuidedElasticaADMMNet-main")
METRICS = {"psnr": True, "ssim": True, "lpips": False,
           "hole_psnr": True, "hole_ssim": True, "hole_l1": False,
           "hole_edge_f1": True, "hole_edge_tolerance_f1": True,
           "hole_gradient_l1": False, "hole_orientation_error": False,
           "hole_inner_psnr": True, "hole_inner_ssim": True, "hole_inner_l1": False}


def path_key(path):
    return str(path).strip().replace("\\", "/").casefold()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_list(path, unique=True):
    lines = [x.strip() for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if not lines or (unique and len({path_key(x) for x in lines}) != len(lines)):
        raise ValueError(f"Empty or duplicate image paths in {path}")
    return lines


def audit_lists(old, v1, override):
    configs = {"old": old, "v1": v1}
    lists, hashes = {}, {}
    for label, cfg in configs.items():
        lists[label] = {}
        for split in ("train", "val"):
            path = cfg["data"][split + "_list"]
            lists[label][split] = read_list(path)
            hashes[label + "_" + split] = {"path": path, "sha256": sha_file(path),
                                            "count": len(lists[label][split])}
    if override:
        selected = read_list(override)
    else:
        if {path_key(x) for x in lists["old"]["val"]} != {path_key(x) for x in lists["v1"]["val"]}:
            raise ValueError("Validation sets differ. Supply --val-list with a shared held-out subset.")
        selected = lists["v1"]["val"]
    keys = {path_key(x) for x in selected}
    # CelebA image names identify samples even if dataset directories were moved.
    names = {path_key(x).rsplit("/", 1)[-1] for x in selected}
    for label in configs:
        train = lists[label]["train"]
        train_names = {path_key(x).rsplit("/", 1)[-1] for x in train}
        if keys.intersection(map(path_key, train)) or names.intersection(train_names):
            raise ValueError(f"Validation/training overlap in {label} (path or image filename).")
        heldout = {path_key(x).rsplit("/", 1)[-1] for x in lists[label]["val"]}
        if not names <= heldout:
            raise ValueError(f"Selected images are not all in {label}'s saved validation list.")
    hashes["selected_order"] = digest_json(selected)
    return selected, hashes


def accept_protocol(out, protocol, resume):
    out = Path(out)
    target = out / "protocol.json"
    if target.exists():
        if not resume:
            raise ValueError(f"Output already exists. Use --resume or a new --output-dir: {out}")
        saved = read_json(target)
        if saved != protocol:
            fields = [k for k in set(saved) | set(protocol) if saved.get(k) != protocol.get(k)]
            raise ValueError(f"Comparison protocol/settings changed: {', '.join(sorted(fields))}. Use a new output directory.")
    else:
        if out.exists() and any(p.name != ".comparison.lock" for p in out.iterdir()):
            raise ValueError(f"Output is nonempty without a protocol: {out}")
        write_json(target, protocol)


@contextmanager
def output_lock(out):
    """OS lock is automatically released even if a process is killed."""
    out.mkdir(parents=True, exist_ok=True)
    handle = (out / ".comparison.lock").open("a+b")
    if handle.seek(0, 2) == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f"Another comparison is using {out}") from None
    try:
        yield
    finally:
        handle.close()


def payload_digest(arrays):
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        array = np.asarray(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_prediction(path, pred, input_id, **extra):
    arrays = {"pred": np.asarray(pred, dtype=np.float32), "input_id": np.asarray(input_id), **extra}
    arrays["payload_hash"] = np.asarray(payload_digest(arrays))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def valid_prediction(path, input_id, size):
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files if name != "payload_hash"}
            return (str(data["input_id"]) == input_id and data["pred"].shape == (3, size, size)
                    and data["pred"].dtype == np.float32 and np.isfinite(data["pred"]).all()
                    and str(data["payload_hash"]) == payload_digest(arrays))
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def input_id(gt, mask):
    return payload_digest({"gt": np.asarray(gt, dtype=np.uint8), "mask": np.asarray(mask, dtype=np.uint8)})


def read_input(out, row):
    import torch
    with Image.open(Path(out) / row["gt_file"]) as image:
        gt = np.asarray(image.convert("RGB")).copy()
    with Image.open(Path(out) / row["mask_file"]) as image:
        mask = np.asarray(image.convert("L")).copy()
    if input_id(gt, mask) != row["input_id"]:
        raise ValueError(f"Prepared image/mask changed for sample {row['index']}")
    return (torch.from_numpy(gt.transpose(2, 0, 1)).float() / 255 * 2 - 1,
            torch.from_numpy(mask[None]).float() / 255)


def code_hashes(project):
    files = sorted((Path(project) / "fg_elastica_inpaint").rglob("*.py"))
    if not files:
        raise FileNotFoundError(f"No fg_elastica_inpaint source under {project}")
    return {str(p.relative_to(project)).replace("\\", "/"): sha_file(p) for p in files}


def prepare_inputs(out, paths, cfg, size, short_side, seed):
    sys.path.insert(0, str(ROOT))
    from fg_elastica_inpaint.data.dataset import _resize_short_side, _center_crop
    from fg_elastica_inpaint.data.mask_generator import RandomMaskGenerator
    from tqdm import tqdm
    data = cfg["data"]
    generator = RandomMaskGenerator(size=size, hole_buckets=data.get("hole_buckets"),
        bucket_probs=data.get("bucket_probs"), topology_modes=data.get("mask_topology_modes"),
        topology_probs=data.get("mask_topology_probs"))
    prior = read_json(out / "inputs/manifest.json") if (out / "inputs/manifest.json").exists() else []
    fixed_path = data.get("val_mask_list")
    fixed = read_list(fixed_path, unique=False) if fixed_path else None
    original_order = read_list(data["val_list"])
    positions = {path_key(p): i for i, p in enumerate(original_order)}
    if fixed is not None and len(fixed) != len(original_order):
        raise ValueError("V1 val_mask_list must have exactly one mask per validation image")
    records = []
    for index, path in enumerate(tqdm(paths, desc="Prepare fixed inputs", dynamic_ncols=True, ascii=True, mininterval=1)):
        source_hash = sha_file(path)
        if index < len(prior):
            row = prior[index]
            if row["source_sha256"] != source_hash or row["path"] != path:
                raise ValueError(f"Source validation image changed: {path}")
            if (out / row["gt_file"]).exists() and (out / row["mask_file"]).exists():
                read_input(out, row)
                records.append(row)
                continue
        with Image.open(path) as image:
            gt = np.asarray(_center_crop(_resize_short_side(image.convert("RGB"), short_side), size)).copy()
        source_index = positions.get(path_key(path))
        if source_index is None:
            raise ValueError(f"Image must use the V1 validation path spelling: {path}")
        random.seed(seed + source_index)
        np.random.seed((seed + source_index) % 2**32)
        if fixed:
            with Image.open(fixed[source_index]) as image:
                values = np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR))
                mask = (values > 127.5).astype(np.uint8) * 255
        else:
            mask = (generator(size, size).squeeze(0).numpy() * 255).astype(np.uint8)
        hole = float((mask == 0).mean())
        if not 0 < hole < 1:
            raise ValueError(f"Sample {index} needs both known and missing pixels")
        row = {"index": index, "source_index": source_index, "path": path, "source_sha256": source_hash,
               "input_id": input_id(gt, mask), "hole_ratio": hole,
               "gt_file": f"inputs/gt/{index:06d}.png", "mask_file": f"inputs/masks/{index:06d}.png"}
        for array, name in ((gt, "gt_file"), (mask, "mask_file")):
            dest = out / row[name]
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name + ".tmp")
            Image.fromarray(array).save(temp, format="PNG")
            os.replace(temp, dest)
        records.append(row)
    write_json(out / "inputs/manifest.json", records)
    (out / "inputs/val.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    (out / "inputs/masks.txt").write_text("\n".join(str(out / r["mask_file"]) for r in records) + "\n", encoding="utf-8")
    return records


def run_child(command, log):
    print("RUN: " + subprocess.list2cmdline(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    # A second project can otherwise be injected into the child through PYTHONPATH.
    env.pop("PYTHONPATH", None)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env)
        # Preserve tqdm carriage returns in the terminal instead of converting
        # each refresh into another line. The UTF-8 log keeps the same output.
        process.stdout.reconfigure(newline="")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        if code:
            raise RuntimeError(f"Child exited with {code}; inspect {log}")


def metric_deltas(old, new):
    rows = []
    for name, higher in METRICS.items():
        a, b = old.get(name), new.get(name)
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            continue
        delta = b - a
        rows.append({"metric": name, "v0": a, "v1": b, "delta_v1_minus_v0": delta,
                     "higher_is_better": higher, "improved": delta > 0 if higher else delta < 0,
                     "relative_improvement_percent": (100 * (a-b) / abs(a)
                         if not higher and abs(a) > 1e-12 else None)})
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def panels(images, names):
    width, height = images[0].size
    canvas = Image.new("RGB", (len(images)*width, height+28), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, name) in enumerate(zip(images, names)):
        canvas.paste(image.convert("RGB"), (i*width, 28))
        draw.text((i*width+8, 7), name, fill="black")
    return canvas


def save_figures(out, row, gt, mask, preds):
    import torch
    from fg_elastica_inpaint.utils.image import tensor_to_pil
    from fg_elastica_inpaint.models.operators import grad
    from fg_elastica_inpaint.utils.structure import edge_strength
    directory = out / "figures" / f"sample_{row['index']:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    completed = [mask*gt + (1-mask)*pred for pred in preds]
    images = [tensor_to_pil(x) for x in [gt*mask, *completed, gt]]
    names = ["Masked", "v0", "V1", "GT"]
    panels(images, names).save(directory / "comparison.png")
    size = gt.shape[-1]
    roi = (int(size*.15), int(size*.2), int(size*.85), int(size*.75))
    crops = [im.crop(roi).resize((size, size), Image.Resampling.NEAREST) for im in images]
    panels(crops, names).save(directory / "central_crop.png")
    edges = [Image.fromarray((edge_strength(grad(x[None]))[0, 0].numpy()*255).round().astype(np.uint8))
             for x in [*completed, gt]]
    panels(edges, ["v0 edge", "V1 edge", "GT edge"]).save(directory / "edges.png")
    with np.load(out / "predictions/v1" / f"{row['index']:06d}.npz", allow_pickle=False) as saved:
        if "gradient" in saved and "edge" in saved:
            g_edge = edge_strength(torch.from_numpy(saved["gradient"])[None])[0, 0].numpy()
            e = saved["edge"][0]
            prior_images = [Image.fromarray((x.clip(0, 1)*255).round().astype(np.uint8)) for x in [e, g_edge]]
            panels([*prior_images, edges[-1]], ["Es", "Edge from Gs", "GT edge"]).save(directory / "structure.png")
    write_json(directory / "metadata.json", {**row, "central_crop_xyxy": roi,
        "display": "PNG clipped to [-1,1]; metrics use original FP32 outputs"})


def score(out, check_only=False, count=None):
    import torch
    from tqdm import tqdm
    sys.path.insert(0, str(ROOT))
    from fg_elastica_inpaint.utils.metrics import (OptionalLPIPS, evaluate_per_image,
        summarize_metric_items, mask_ratio_bucket, metric_sample_supported, MASK_RATIO_BUCKETS)
    protocol = read_json(out / "protocol.json")
    torch.set_num_threads(1)
    device = torch.device(protocol["device"])
    lpips = None if protocol["no_lpips"] else OptionalLPIPS(device)
    if lpips is not None and not lpips.ready:
        raise RuntimeError("LPIPS unavailable: " + str(lpips.unavailable_reason)
                           + ". Install/cache LPIPS in the GPU environment, or explicitly choose --no-lpips with a new output directory.")
    print(f"Scoring device={device}; LPIPS={'disabled (not reported)' if lpips is None else 'AlexNet ready'}", flush=True)
    if check_only:
        return
    manifest = read_json(out / "inputs/manifest.json")
    if count is not None:
        manifest = manifest[:count]
    results = {"old": [], "v1": []}
    score_signature = digest_json(protocol)
    with torch.inference_mode():
        for row in tqdm(manifest, desc="Unified scoring", dynamic_ncols=True, ascii=True, mininterval=1):
            gt, mask = read_input(out, row)
            preds = []
            for label in results:
                path = out / "predictions" / label / f"{row['index']:06d}.npz"
                if not valid_prediction(path, row["input_id"], protocol["image_size"]):
                    raise ValueError(f"Missing/corrupt prediction: {path}; rerun with --resume")
                with np.load(path, allow_pickle=False) as data:
                    pred = torch.from_numpy(data["pred"].copy())
                    prediction_hash = str(data["payload_hash"])
                preds.append(pred)
                cache = out / "scores" / label / f"{row['index']:06d}.json"
                previous = read_json(cache) if cache.exists() else {}
                if (previous.get("signature") == score_signature and previous.get("prediction_hash") == prediction_hash):
                    item = previous["metrics"]
                else:
                    item = evaluate_per_image(pred[None].to(device), gt[None].to(device), mask[None].to(device), lpips)[0]
                    if not all(math.isfinite(float(v)) for v in item.values()):
                        raise ValueError(f"Nonfinite metrics for {label} sample {row['index']}")
                    write_json(cache, {"signature": score_signature, "prediction_hash": prediction_hash, "metrics": item})
                results[label].append(item)
            if row["index"] < protocol["visual_count"]:
                save_figures(out, row, gt, mask, preds)
    report_dir = out / ("preflight_report" if count is not None else "report")
    summaries, comparisons = {}, []
    buckets = ["all", *[b for b, _, _ in MASK_RATIO_BUCKETS]]
    for label, items in results.items():
        named = [{**{k: r[k] for k in ("index", "path", "input_id", "source_sha256")}, **item,
                  "index": r["index"], "bucket": mask_ratio_bucket(item["hole_ratio"])} for r, item in zip(manifest, items)]
        write_csv(report_dir / f"{label}_per_image.csv", named)
        summaries[label] = {}
        for bucket in buckets:
            selected = [x for x in items if bucket == "all" or mask_ratio_bucket(x["hole_ratio"]) == bucket]
            summaries[label][bucket] = {"num_samples": len(selected), **summarize_metric_items(selected)}
        write_csv(report_dir / f"{label}_buckets.csv", [{"bucket": b, **v} for b, v in summaries[label].items()])
    for bucket in buckets:
        comparisons.extend({"bucket": bucket, "num_samples": summaries["old"][bucket]["num_samples"], **x}
                           for x in metric_deltas(summaries["old"][bucket], summaries["v1"][bucket]))
    write_csv(report_dir / "comparison.csv", comparisons)
    pairs, counts = [], {}
    for row, old, new in zip(manifest, results["old"], results["v1"]):
        entry = {"index": row["index"], "path": row["path"], "input_id": row["input_id"],
                 "hole_ratio": row["hole_ratio"]}
        for delta in metric_deltas(old, new):
            name = delta["metric"]
            if not metric_sample_supported(old, name) or not metric_sample_supported(new, name):
                continue
            entry[name + "_delta"] = delta["delta_v1_minus_v0"]
            counter = counts.setdefault(name, {"improved": 0, "worse": 0, "tied": 0, "supported": 0})
            counter["supported"] += 1
            counter["tied" if abs(delta["delta_v1_minus_v0"]) <= 1e-10 else "improved" if delta["improved"] else "worse"] += 1
        pairs.append(entry)
    write_csv(report_dir / "per_image_deltas.csv", pairs)
    write_json(report_dir / "results.json", {"summary": summaries, "counts": counts, "num_samples": len(manifest),
        "scope": "Historical v0 vs full V1. Not an isolated Structure Prior training ablation.",
        "metrics": "Shared current implementation; FP32 outputs in [-1,1] units without display clipping; LPIPS on completed RGB."})
    model_info = {label: read_json(out / "protocol" / f"{label}_model.json") for label in results}
    write_json(report_dir / "models.json", model_info)
    lines = ["# v0 / V1 comparison", "", f"Samples: {len(manifest)}. Same prepared RGB and mask hashes for both models.", "",
             "Historical model comparison, not an isolated Structure Prior ablation. Different model/training budgets remain in place.", "",
             "| Metric | v0 | V1 | V1 - v0 |", "|---|---:|---:|---:|"]
    for delta in comparisons:
        if delta["bucket"] == "all":
            lines.append(f"| {delta['metric']} | {delta['v0']:.6f} | {delta['v1']:.6f} | {delta['delta_v1_minus_v0']:+.6f} |")
    lines += ["", "LPIPS disabled; do not report a value." if lpips is None else "LPIPS: AlexNet, completed RGB.", "",
              "Per-bucket results: comparison.csv. Per-case improvements: per_image_deltas.csv.",
              "Figures are the first preselected validation samples, not selected by improvement.",
              "Central crops use the same fixed coordinates for both outputs; they are not detected facial landmarks."]
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {report_dir}", flush=True)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] in ("--internal-score", "--internal-check-metrics"):
        score(Path(sys.argv[2]), check_only=sys.argv[1] == "--internal-check-metrics",
              count=int(sys.argv[3]) if len(sys.argv) > 3 else None)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-project", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--v1-project", type=Path, default=ROOT)
    for label in ("old", "v1"):
        parser.add_argument(f"--{label}-checkpoint", type=Path)
        parser.add_argument(f"--{label}-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("C:/codeee/inpainting/comparison_v0_v1"))
    parser.add_argument("--val-list", type=Path, help="Optional shared held-out subset; use V1 image path spellings")
    parser.add_argument("--limit", type=int, help="First N shared images; use a separate output directory for a small run")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--visual-count", type=int, default=16)
    parser.add_argument("--preflight-count", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-lpips", action="store_true", help="Explicitly omit LPIPS (not a zero score)")
    args = parser.parse_args()
    if min(args.batch_size, args.preflight_count) < 1 or args.visual_count < 0 or (args.limit is not None and args.limit < 1):
        parser.error("batch/preflight/limit must be positive and visual-count nonnegative")
    import torch
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    out = args.output_dir.resolve()
    models, configs = {}, {}
    for label in ("old", "v1"):
        project = getattr(args, label + "_project").resolve()
        default_run = project / ("outputs/fg_elastica_full_21p" if label == "old" else "outputs/structure_v1/r2")
        checkpoint = (getattr(args, label + "_checkpoint") or default_run / "best.pt").resolve()
        config = (getattr(args, label + "_config") or default_run / ("config_resolved.yaml" if label == "old" else "runner_config.yaml")).resolve()
        configs[label] = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
        models[label] = {"project": str(project), "checkpoint_source": str(checkpoint),
                         "checkpoint_sha256": sha_file(checkpoint), "config": configs[label],
                         "config_source": str(config), "code": code_hashes(project)}
    paths, list_audit = audit_lists(configs["old"], configs["v1"], args.val_list)
    for field, default in (("image_size", 256), ("resize_short_to", 286)):
        a, b = (int(configs[label]["data"].get(field, default)) for label in ("old", "v1"))
        if a != b:
            raise ValueError(f"Preprocessing differs: {field} old={a}, V1={b}. Decide a shared evaluation transform explicitly first.")
    size = int(configs["v1"]["data"].get("image_size", 256))
    short_side = int(configs["v1"]["data"].get("resize_short_to", 286))
    if args.limit:
        paths = paths[:args.limit]
    fixed_list = configs["v1"]["data"].get("val_mask_list")
    fixed_hashes = [sha_file(p) for p in read_list(fixed_list, unique=False)] if fixed_list else None
    packages = {}
    for name in ("torch", "numpy", "Pillow", "lpips", "torchvision"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    protocol = {"version": 1, "models": models, "lists": list_audit, "paths": paths,
        "image_size": size, "resize_short_to": short_side, "mask_seed": configs["v1"]["data"].get("eval_mask_seed", 1701),
        "fixed_mask_hashes": fixed_hashes, "device": device, "batch_size": args.batch_size,
        "visual_count": args.visual_count, "no_lpips": args.no_lpips, "packages": packages,
        "scorer_code": code_hashes(ROOT), "tool_code": {p.name: sha_file(p) for p in
            (Path(__file__), ROOT / "tools/comparison_worker.py")},
        "metric_options": {"edge_scale": .1, "edge_threshold": .5, "orientation_threshold": .05,
                           "edge_tolerance": 1, "boundary_width": 4}}
    with output_lock(out):
        accept_protocol(out, protocol, args.resume)
        execute(args, out, protocol, paths, configs, models)


def execute(args, out, protocol, paths, configs, models):
    size, short_side, device = protocol["image_size"], protocol["resize_short_to"], protocol["device"]
    print(f"Comparison: {len(paths)} validation images | {size}px | batch={args.batch_size} | {device}", flush=True)
    print("Source checkpoints and model code remain unchanged. Resume uses validated cached FP32 predictions.", flush=True)
    worker = ROOT / "tools/comparison_worker.py"
    for label, model in models.items():
        snapshot = out / "protocol" / label / "checkpoint.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists() or sha_file(snapshot) != model["checkpoint_sha256"]:
            temporary = snapshot.with_suffix(".tmp")
            shutil.copyfile(model["checkpoint_source"], temporary)
            if sha_file(temporary) != model["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed while copying; wait for training/save to finish")
            os.replace(temporary, snapshot)
        write_json(out / "protocol" / label / "config.json", model["config"])
        run_child([sys.executable, "-B", str(worker), "--output", str(out), "--model", label, "--inspect"], out / f"{label}.log")
    run_child([sys.executable, "-B", str(Path(__file__).resolve()), "--internal-check-metrics", str(out)], out / "scoring.log")
    prepare_inputs(out, paths, configs["v1"], size, short_side, protocol["mask_seed"])
    for label in models:
        run_child([sys.executable, "-B", str(worker), "--output", str(out), "--model", label,
                   "--count", str(args.preflight_count)], out / f"{label}.log")
    run_child([sys.executable, "-B", str(Path(__file__).resolve()), "--internal-score", str(out), str(args.preflight_count)], out / "scoring.log")
    if args.preflight_only:
        print("Preflight complete. Run the same command with --resume and without --preflight-only to evaluate all images.")
        return
    for label in models:
        run_child([sys.executable, "-B", str(worker), "--output", str(out), "--model", label], out / f"{label}.log")
    run_child([sys.executable, "-B", str(Path(__file__).resolve()), "--internal-score", str(out)], out / "scoring.log")
    print(f"DONE: {out / 'report/README.md'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Repeat with --resume to continue.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Comparison error: {error}", file=sys.stderr)
        sys.exit(1)
