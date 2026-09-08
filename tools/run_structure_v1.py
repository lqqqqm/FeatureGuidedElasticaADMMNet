"""Sequential V1 training/evaluation using the existing project entry points.

Run from any directory. Relative paths are resolved against the project root.
No packages are installed and no model equations are changed by this launcher.
"""
from __future__ import annotations

import argparse
import codecs
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ("r0", "r1", "r2")
PRIOR_LOSSES = ("lambda_structure_grad", "lambda_structure_edge",
                "lambda_structure_orientation", "lambda_structure_consistency")
LIST_KEYS = ("train_list", "val_list", "test_list", "val_mask_list", "test_mask_list")


def resolve(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def load_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS))
    p.add_argument("--config-dir", default=str(ROOT/"configs"))
    p.add_argument("--output-dir", default=str(ROOT/"outputs/structure_v1"))
    for key in LIST_KEYS:
        p.add_argument("--"+key.replace("_", "-"), help="Override this file list identically in selected runs")
    p.add_argument("--batch-size", type=int, help="Override train/val/test batch sizes together")
    p.add_argument("--epochs", type=int, help="Override the training budget before starting a new experiment")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--checkpoint", choices=("best.pt", "best_structure.pt", "last.pt"), default="best.pt")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Resume interrupted runs and skip unchanged completed evaluations")
    mode.add_argument("--eval-only", action="store_true", help="Re-evaluate existing launcher runs without training")
    p.add_argument("--preflight", action="store_true", help="Run training-only numerical preflight once before training; report only, no automatic tuning")
    p.add_argument("--preflight-samples", type=int, default=32)
    p.add_argument("--dry-run", action="store_true", help="Print configs/commands only; no file writes, GPU or dataset access")
    return p


def validate_matched(configs):
    common = None
    supervised_weights = None
    for run, cfg in configs.items():
        model, loss = cfg["model"], cfg["loss"]
        enabled, rho = model.get("use_structure_prior", False), model.get("structure_rho", 0)
        if enabled != (run != "r0") or not math.isfinite(rho) or (rho <= 0 if run == "r2" else rho != 0):
            raise ValueError(f"{run}: expected R0=no prior, R1=prior/rho=0, R2=prior/rho>0")
        if not model.get("use_unrolling", True) or not model.get("use_pcg_readout", False):
            raise ValueError(f"{run}: V1 requires ADMM unrolling and PCG readout")
        if model.get("enable_p_correction", False) or model.get("enable_n_correction", False):
            raise ValueError(f"{run}: V1 controls require p/n Correction disabled")
        weights = [loss.get(k, 0) for k in PRIOR_LOSSES]
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError(f"{run}: invalid structure loss weights")
        if (run == "r0" and any(weights)) or (run != "r0" and weights[0] <= 0):
            raise ValueError(f"{run}: incorrect explicit structure supervision for this control")
        if run != "r0":
            if supervised_weights is not None and weights != supervised_weights:
                raise ValueError("R1 and R2 structure supervision weights must match")
            supervised_weights = weights
        reduced = copy.deepcopy(cfg)
        reduced.pop("run_name", None)
        reduced.pop("output_dir", None)
        for key in ("use_structure_prior", "structure_rho"):
            reduced["model"].pop(key, None)
        for key in PRIOR_LOSSES:
            reduced["loss"].pop(key, None)
        if common is not None and reduced != common:
            raise ValueError("Selected/existing V1 runs have mismatched data, model, solver, loss or training settings")
        common = reduced


def make_configs(args):
    if len(set(args.runs)) != len(args.runs):
        raise ValueError("--runs must not contain duplicates")
    for key in ("epochs", "batch_size", "preflight_samples"):
        value = getattr(args, key)
        if value is not None and value < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.eval_only and args.preflight:
        raise ValueError("--eval-only cannot be combined with --preflight")
    output = resolve(args.output_dir)
    configs = {}
    for run in args.runs:
        cfg = load_yaml(resolve(args.config_dir)/f"structure_v1_{run}.yaml")
        data = cfg["data"]
        for key in LIST_KEYS:
            value = getattr(args, key) or data.get(key)
            data[key] = str(resolve(value)) if value else None
        if args.batch_size is not None:
            for key in ("batch_size", "val_batch_size", "test_batch_size"):
                data[key] = args.batch_size
        if args.num_workers is not None:
            data["num_workers"] = args.num_workers
        if args.epochs is not None:
            cfg["optim"]["epochs"] = args.epochs
        cfg.update(output_dir=str(output), run_name=run)
        configs[run] = cfg
    validate_matched(configs)
    return output, configs


def check_data(cfg):
    """Check all listed paths once, without decoding images or reading GT pixels."""
    sets, fingerprints = {}, {}
    for key in LIST_KEYS:
        path = cfg["data"].get(key)
        if not path:
            if key in ("train_list", "val_list", "test_list"):
                raise ValueError(f"data.{key} is required for train/validation/test evaluation")
            continue
        content = Path(path).read_bytes()
        if content.startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"Save the file list as UTF-8 without BOM (required by the dataset reader): {path}")
        entries = [line.strip() for line in content.decode("utf-8").splitlines() if line.strip()]
        if not entries:
            raise ValueError(f"Empty file list: {path}")
        paths = set()
        for index, entry in enumerate(entries, 1):
            # Match the dataset reader: it opens each literal path without expanding ~.
            image = Path(entry)
            if not image.is_absolute():
                raise ValueError(f"Use absolute image paths in {path}, line {index}: {entry}")
            if not image.is_file():
                raise FileNotFoundError(f"Missing image/mask in {path}, line {index}: {image}")
            paths.add(image.resolve())
        fingerprints[key] = hashlib.sha256(content).hexdigest()
        sets[key] = paths
        if key == "train_list":
            limit = cfg["data"].get("train_limit")
            size = min(len(entries), int(limit)) if limit is not None else len(entries)
            if size < cfg["data"].get("batch_size", 4):
                raise ValueError("Training set/limit is smaller than batch_size (drop_last would produce zero batches)")
    for first, second in (("train_list", "val_list"), ("train_list", "test_list"), ("val_list", "test_list")):
        if sets[first] & sets[second]:
            raise ValueError(f"Image overlap between {first} and {second}; use disjoint splits")
    return fingerprints


def check_device(device):
    if device == "cpu":
        print("Device: CPU (explicitly requested)", flush=True)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch environment; use --device cpu only for a small local check")
    torch.ones(1, device="cuda").add_(1).cpu()
    print(f"Device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)


@contextmanager
def experiment_lock(output):
    """OS lock is released even after a crash; the small lock file may remain."""
    output.mkdir(parents=True, exist_ok=True)
    with (output/".runner.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another runner is using {output}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def describe_differences(saved, current, prefix):
    """Report exact resume mismatches without changing the acceptance criteria."""
    if isinstance(saved, dict) and isinstance(current, dict):
        lines = []
        for key in sorted(saved.keys() | current.keys()):
            name = f"{prefix}.{key}"
            if key not in saved:
                lines.append(f"{name}: saved=<missing>; current={current[key]!r}")
            elif key not in current:
                lines.append(f"{name}: saved={saved[key]!r}; current=<missing>")
            else:
                lines.extend(describe_differences(saved[key], current[key], name))
        return lines
    return [f"{prefix}: saved={saved!r}; current={current!r}"] if saved != current else []


def check_existing(output, configs, args, fingerprints):
    combined = dict(configs)
    for run in RUNS:
        saved = output/run/"runner_config.yaml"
        if run not in configs and saved.is_file():
            combined[run] = load_yaml(saved)
            state = json.loads((output/run/"runner_state.json").read_text(encoding="utf-8"))
            if state.get("data_fingerprints") != fingerprints:
                details = describe_differences(state.get("data_fingerprints"), fingerprints, "list_sha256")
                raise ValueError(f"Data lists changed since existing {run}:\n  " + "\n  ".join(details)
                                 + "\nRestore the original lists or use a separate output directory.")
    validate_matched(combined)
    for run, cfg in configs.items():
        directory = output/run
        if not directory.exists() or not any(directory.iterdir()):
            if args.eval_only:
                raise FileNotFoundError(f"No existing run to evaluate: {directory}")
            continue
        if not (args.resume or args.eval_only):
            raise FileExistsError(f"Existing experiment: {directory}. Use --resume or a new --output-dir")
        if not (directory/"runner_config.yaml").is_file() or not (directory/"runner_state.json").is_file():
            raise ValueError(f"Not a launcher-managed run: {directory}; use train.py/evaluate.py directly")
        state = json.loads((directory/"runner_state.json").read_text(encoding="utf-8"))
        differences = describe_differences(load_yaml(directory/"runner_config.yaml"), cfg, "config")
        differences.extend(describe_differences(state.get("data_fingerprints"), fingerprints, "list_sha256"))
        if differences:
            raise ValueError(f"Config or data lists changed for {run}:\n  " + "\n  ".join(differences)
                             + f"\nSaved config: {directory/'runner_config.yaml'}"
                             + f"\nSaved list hashes: {directory/'runner_state.json'}"
                             + "\nList hashes include contents, order, encoding and line endings (not timestamps)."
                             + "\nRestore the listed settings/files or use a new output directory; training was not started.")
        if args.eval_only and not (directory/args.checkpoint).is_file():
            raise FileNotFoundError(f"Missing evaluation checkpoint: {directory/args.checkpoint}")


class ProgressRelay:
    """Preserve terminal redraws while writing occasional plain progress lines."""

    def __init__(self, console, log):
        self.console, self.log = console, log
        self.interactive = console.isatty()
        self.buffer = []
        self.progress = False
        self.pending_cr = False
        self.last_progress_time = -float("inf")
        self.last_progress_line = None

    def _record(self, final=False):
        line = "".join(self.buffer).rstrip()
        self.buffer.clear()
        if not line:
            return
        if self.progress:
            now = time.monotonic()
            if line == self.last_progress_line or (not final and now - self.last_progress_time < 30):
                return
            self.last_progress_time, self.last_progress_line = now, line
        self.log.write(line + "\n")
        self.log.flush()
        if not self.interactive:
            self.console.write(line + "\n")
            self.console.flush()

    def feed(self, text):
        if self.interactive:
            self.console.write(text)
            self.console.flush()
        for char in text:
            # Delay a CR boundary by one character so Windows CRLF is treated
            # as a completed line, including the final progress snapshot.
            if self.pending_cr:
                self._record(final=char == "\n")
                self.progress = char != "\n"
                self.pending_cr = False
                if char == "\n":
                    continue
            if char == "\r":
                self.pending_cr = True
            elif char == "\n":
                self._record(final=True)
                self.progress = False
            else:
                self.buffer.append(char)

    def finish(self):
        if self.interactive and (self.buffer or self.progress or self.pending_cr):
            self.console.write("\n")
            self.console.flush()
        self._record(final=True)


def run_command(command, logfile, device):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["FG_ELASTICA_PROGRESS"] = "1"
    env["COLUMNS"] = str(shutil.get_terminal_size(fallback=(110, 30)).columns)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    print("COMMAND: "+subprocess.list2cmdline(command), flush=True)
    with logfile.open("a", encoding="utf-8") as log:
        log.write("\n"+time.strftime("%Y-%m-%d %H:%M:%S")+" "+subprocess.list2cmdline(command)+"\n")
        log.flush()
        relay = ProgressRelay(sys.stdout, log)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        # Binary reads preserve tqdm's carriage returns; text=True would turn
        # them into newlines before the console ever sees them.
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as child:
            try:
                while chunk := child.stdout.read1(4096):
                    relay.feed(decoder.decode(chunk))
                relay.feed(decoder.decode(b"", final=True))
                code = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                raise
            finally:
                relay.finish()
        if code:
            raise RuntimeError(f"Command failed with exit code {code}; see {logfile}")


def checkpoint_signature(path):
    info = path.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def write_summary(output):
    report = {}
    rows = []
    for run in RUNS:
        path = output/run/"runner_state.json"
        if not path.is_file():
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        result = {"status": state.get("status"), "error": state.get("error"),
                  "config": str(output/run/"runner_config.yaml")}
        selected = state.get("selected_checkpoint", "best.pt")
        result["checkpoint"] = str(output/run/selected)
        evaluation = state.get("evaluations", {}).get(selected)
        if evaluation:
            result["metrics"] = evaluation["metrics"]
        report[run] = result
        rows.append({"run": run, **{k: v for k, v in result.items() if k != "metrics"}, **result.get("metrics", {})})
    save_json(output/"summary.json", report)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output/"summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)


def execute_run(run, cfg, output, args, fingerprints):
    directory = output/run
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory/"runner_config.yaml"
    state_path = directory/"runner_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        state = {"data_fingerprints": fingerprints, "training_complete": False, "evaluations": {}}
    state.update(selected_checkpoint=args.checkpoint, error=None)
    save_json(state_path, state)
    try:
        if not args.eval_only and not state["training_complete"]:
            command = [sys.executable, "-u", str(ROOT/"train.py"), "--config", str(config_path)]
            if args.resume and (directory/"last.pt").is_file():
                command += ["--resume", str(directory/"last.pt")]
            state["status"] = "training"
            save_json(state_path, state)
            run_command(command, directory/"train.log", args.device)
            state.update(training_complete=True, status="trained")
            save_json(state_path, state)
        checkpoint = directory/args.checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training did not produce {checkpoint}")
        signature = checkpoint_signature(checkpoint)
        previous = state["evaluations"].get(args.checkpoint)
        csv_path = directory/"eval_test_per_image.csv"
        csv_snapshot = directory/f"eval_test_{checkpoint.stem}_per_image.csv"
        if args.resume and previous and previous.get("checkpoint_signature") == signature and csv_snapshot.is_file():
            # Switching checkpoints must also switch the common evaluation artifacts.
            save_json(directory/"eval_test.json", previous["metrics"])
            shutil.copyfile(csv_snapshot, csv_path)
            state["status"] = "complete"
            print(f"[{run}] Training and evaluation already complete; skipping", flush=True)
        else:
            state["status"] = "evaluating"
            state["evaluations"].pop(args.checkpoint, None)
            save_json(state_path, state)
            result_path = directory/"eval_test.json"
            if result_path.exists():
                result_path.replace(directory/"eval_test_previous.json")
            if csv_path.exists():
                csv_path.replace(directory/"eval_test_previous_per_image.csv")
            run_command([sys.executable, "-u", str(ROOT/"evaluate.py"), "--config", str(config_path),
                         "--checkpoint", str(checkpoint), "--split", "test"], directory/"evaluate.log", args.device)
            metrics = json.loads(result_path.read_text(encoding="utf-8"))
            for key in ("psnr", "ssim", "hole_psnr"):
                if key not in metrics or not isinstance(metrics[key], (int, float)) or not math.isfinite(metrics[key]):
                    raise ValueError(f"Evaluation is missing a finite {key}: {directory/'eval_test.json'}")
            if not csv_path.is_file():
                raise FileNotFoundError(f"Evaluation did not produce {csv_path}")
            shutil.copyfile(csv_path, csv_snapshot)
            state["evaluations"][args.checkpoint] = {"checkpoint_signature": signature, "metrics": clean_json(metrics)}
            save_json(directory/f"eval_test_{checkpoint.stem}.json", metrics)
            state["status"] = "complete"
        save_json(state_path, state)
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        save_json(state_path, state)
        raise
    finally:
        write_summary(output)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        output, configs = make_configs(args)
        if args.preflight:
            print("Preflight before training: "+subprocess.list2cmdline([
                sys.executable, "-u", str(ROOT/"tools/preflight_structure_v1.py"), "--config", str(output/"preflight_config.yaml"),
                "--samples", str(args.preflight_samples), "--device", args.device, "--output", str(output/"preflight.json")]))
        for run, cfg in configs.items():
            path = output/run/"runner_config.yaml"
            print(f"[{run}] {cfg['data']['image_size']}px | batch={cfg['data']['batch_size']} | epochs={cfg['optim']['epochs']} | rho={cfg['model'].get('structure_rho', 0)}")
            print(f"  config: {path}")
            print(f"  train: {cfg['data'].get('train_list')}\n  val: {cfg['data'].get('val_list')}\n  test: {cfg['data'].get('test_list')}")
            command = [sys.executable, "-u", str(ROOT/"train.py"), "--config", str(path)]
            if args.resume and (output/run/"last.pt").is_file():
                command += ["--resume", str(output/run/"last.pt")]
            if not args.eval_only:
                print("  "+subprocess.list2cmdline(command))
            print("  "+subprocess.list2cmdline([sys.executable, "-u", str(ROOT/"evaluate.py"), "--config", str(path),
                                               "--checkpoint", str(output/run/args.checkpoint), "--split", "test"]))
        if args.dry_run:
            print("Dry run: no files written, no data/GPU validation, no processes launched")
            return 0
        fingerprints = check_data(next(iter(configs.values())))
        check_existing(output, configs, args, fingerprints)
        check_device(args.device)
        with experiment_lock(output):
            check_existing(output, configs, args, fingerprints)
            if args.preflight:
                source = "r2" if "r2" in configs else args.runs[0]
                path = output/"preflight_config.yaml"
                path.write_text(yaml.safe_dump(configs[source], sort_keys=False), encoding="utf-8")
                run_command([sys.executable, "-u", str(ROOT/"tools/preflight_structure_v1.py"), "--config", str(path),
                             "--samples", str(args.preflight_samples), "--device", args.device,
                             "--output", str(output/"preflight.json")], output/"preflight.log", args.device)
                print("Preflight report saved. Configured hyperparameters are unchanged; recommendations are not applied automatically.", flush=True)
            for run, cfg in configs.items():
                execute_run(run, cfg, output, args, fingerprints)
        print(f"Finished. Summary: {output/'summary.csv'}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Re-run with --resume and the same settings to continue.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, yaml.YAMLError) as exc:
        print(f"V1 runner error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
