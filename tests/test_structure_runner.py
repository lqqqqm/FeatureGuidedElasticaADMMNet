"""Exercise orchestration with real tiny child processes, without GPU training."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/run_structure_v1.py"

# The fixture processes implement train/evaluate's file-and-exit-code contract.
CHILD = '''
import argparse, json
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--checkpoint')
p.add_argument('--split')
p.add_argument('--resume')
a=p.parse_args()
c=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
d=Path(c['output_dir'])/c['run_name'];d.mkdir(parents=True,exist_ok=True)
kind=Path(__file__).stem
with Path('events.jsonl').open('a',encoding='utf-8') as f:
 f.write(json.dumps({'kind':kind,'run':c['run_name'],'resume':a.resume})+'\\n')
print(kind+' '+c['run_name'],flush=True)
if kind=='train':
 (d/'last.pt').write_text('last',encoding='utf-8')
 if Path('fail_'+c['run_name']).exists(): raise SystemExit(7)
 (d/'best.pt').write_text('best',encoding='utf-8')
else:
 assert Path(a.checkpoint).is_file()
 if Path('omit_metrics').exists(): raise SystemExit(0)
 score=22.5 if Path(a.checkpoint).name=='last.pt' else 21.5
 (d/'eval_test.json').write_text(json.dumps({'psnr':score,'ssim':.7,'hole_psnr':18.,'lpips_available':0,'fid':float('nan')}),encoding='utf-8')
 (d/'eval_test_per_image.csv').write_text('checkpoint,psnr\\n'+Path(a.checkpoint).name+','+str(score)+'\\n',encoding='utf-8')
'''


class StructureRunnerTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.tmp = tempfile.TemporaryDirectory(prefix="structure runner ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        self.output = self.root / "experiment results"
        lists = {}
        for split in ("train", "val", "test"):
            images = []
            for i in range(4 if split == "train" else 1):
                image = self.root/f"{split} face {i}.png"
                image.write_bytes(b"only file existence is checked by the launcher")
                images.append(str(image))
            path = self.root/f"{split} list.txt"
            path.write_text("\n".join(images), encoding="utf-8")
            lists[split+"_list"] = str(path)
        for name in ("r0", "r1", "r2"):
            cfg = yaml.safe_load((ROOT/f"configs/structure_v1_{name}.yaml").read_text(encoding="utf-8"))
            cfg["data"].update(lists)
            (self.config_dir/f"structure_v1_{name}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        for script in ("train.py", "evaluate.py"):
            (self.root/script).write_text(CHILD, encoding="utf-8")

    def runner(self):
        self.assertTrue(SCRIPT.is_file(), "The V1 runner is not implemented")
        spec = importlib.util.spec_from_file_location("structure_runner", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = self.root
        return module

    def arguments(self, *extra):
        return ["--config-dir", str(self.config_dir), "--output-dir", str(self.output),
                "--device", "cpu", *extra]

    def events(self):
        path = self.root/"events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_dry_run_has_no_processes_or_output_writes(self):
        runner = self.runner()
        self.assertEqual(runner.main(self.arguments("--dry-run")), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.events(), [])

    def test_three_runs_train_then_evaluate_and_keep_unavailable_metrics_missing(self):
        runner = self.runner()
        original = (self.config_dir/"structure_v1_r2.yaml").read_bytes()
        self.assertEqual(runner.main(self.arguments()), 0)
        self.assertEqual([(e["run"], e["kind"]) for e in self.events()],
                         [(r, kind) for r in ("r0", "r1", "r2") for kind in ("train", "evaluate")])
        report = json.loads((self.output/"summary.json").read_text(encoding="utf-8"))
        self.assertEqual(set(report), {"r0", "r1", "r2"})
        self.assertEqual(report["r2"]["metrics"]["hole_psnr"], 18.)
        self.assertNotIn("lpips", report["r2"]["metrics"])
        self.assertIsNone(report["r2"]["metrics"]["fid"])
        self.assertTrue((self.output/"summary.csv").is_file())
        self.assertIn("train r2", (self.output/"r2/train.log").read_text())
        self.assertEqual(original, (self.config_dir/"structure_v1_r2.yaml").read_bytes())

    def test_failure_stops_later_runs_and_resume_does_not_repeat_completed_work(self):
        runner = self.runner()
        failure = self.root/"fail_r1"
        failure.touch()
        self.assertNotEqual(runner.main(self.arguments()), 0)
        self.assertEqual([(e["run"], e["kind"]) for e in self.events()],
                         [("r0", "train"), ("r0", "evaluate"), ("r1", "train")])
        failure.unlink()
        self.assertEqual(runner.main(self.arguments("--resume")), 0)
        events = self.events()
        self.assertEqual(sum(e["run"] == "r0" for e in events), 2)
        resumed = [e for e in events if e["run"] == "r1" and e["resume"]]
        self.assertEqual(len(resumed), 1)
        self.assertEqual(Path(resumed[0]["resume"]), self.output/"r1/last.pt")
        self.assertEqual(events[-1]["run"], "r2")

    def test_existing_experiment_requires_resume_and_changed_config_is_rejected(self):
        runner = self.runner()
        self.assertEqual(runner.main(self.arguments("--runs", "r2")), 0)
        before = self.events()
        self.assertNotEqual(runner.main(self.arguments("--runs", "r2")), 0)
        self.assertNotEqual(runner.main(self.arguments("--runs", "r2", "--resume", "--epochs", "2")), 0)
        self.assertEqual(before, self.events())

    def test_resume_error_identifies_changed_config_fields(self):
        runner = self.runner()
        self.assertEqual(runner.main(self.arguments("--runs", "r2")), 0)
        args = runner.parser().parse_args(self.arguments("--runs", "r2", "--resume", "--epochs", "2"))
        output, configs = runner.make_configs(args)
        with self.assertRaisesRegex(ValueError, r"config.optim.epochs: saved=40; current=2"):
            runner.check_existing(output, configs, args, runner.check_data(configs["r2"]))

    def test_resume_error_identifies_changed_data_list_separately(self):
        runner = self.runner()
        self.assertEqual(runner.main(self.arguments("--runs", "r2")), 0)
        path = self.root/"train list.txt"
        path.write_text("\n".join(reversed(path.read_text().splitlines())), encoding="utf-8")
        args = runner.parser().parse_args(self.arguments("--runs", "r2", "--resume"))
        output, configs = runner.make_configs(args)
        with self.assertRaisesRegex(ValueError, r"list_sha256.train_list: saved="):
            runner.check_existing(output, configs, args, runner.check_data(configs["r2"]))

    def test_resume_accepts_code_changes_and_equivalent_cli_path_separators(self):
        runner = self.runner()
        lists = [flag for split in ("train", "val", "test")
                 for flag in ("--"+split+"-list", str(self.root/f"{split} list.txt"))]
        self.assertEqual(runner.main(self.arguments("--runs", "r2", *lists)), 0)
        (self.root/"train.py").write_text(CHILD+"\n# Training-loop fix only\n", encoding="utf-8")
        lists = [flag for split in ("train", "val", "test")
                 for flag in ("--"+split+"-list", (self.root/f"{split} list.txt").as_posix())]
        before = self.events()
        self.assertEqual(runner.main(self.arguments("--runs", "r2", "--resume", *lists)), 0)
        self.assertEqual(before, self.events())

    def test_mismatched_experiments_and_missing_images_fail_before_training(self):
        runner = self.runner()
        path = self.config_dir/"structure_v1_r1.yaml"
        cfg = yaml.safe_load(path.read_text()); cfg["model"]["K"] = 99
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        self.assertNotEqual(runner.main(self.arguments()), 0)
        self.assertFalse(self.output.exists())
        (self.root/"test face 0.png").unlink()
        self.assertNotEqual(runner.main(self.arguments("--runs", "r2")), 0)
        self.assertEqual(self.events(), [])

    def test_evaluation_only_reuses_saved_model_without_training(self):
        runner = self.runner()
        self.assertEqual(runner.main(self.arguments("--runs", "r2")), 0)
        self.assertEqual(runner.main(self.arguments("--runs", "r2", "--eval-only")), 0)
        self.assertEqual([e["kind"] for e in self.events()], ["train", "evaluate", "evaluate"])

    def test_changed_lists_cannot_be_mixed_with_an_existing_unselected_run(self):
        runner = self.runner()
        self.assertEqual(runner.main(self.arguments("--runs", "r0")), 0)
        path = self.root/"train list.txt"
        path.write_text("\n".join(reversed(path.read_text().splitlines())), encoding="utf-8")
        self.assertNotEqual(runner.main(self.arguments("--runs", "r2")), 0)
        self.assertFalse((self.output/"r2").exists())

    def test_failed_reevaluation_cannot_reuse_old_metric_files(self):
        runner = self.runner()
        self.assertEqual(runner.main(self.arguments("--runs", "r2")), 0)
        (self.root/"omit_metrics").touch()
        self.assertNotEqual(runner.main(self.arguments("--runs", "r2", "--eval-only")), 0)
        report = json.loads((self.output/"summary.json").read_text())
        self.assertEqual(report["r2"]["status"], "failed")
        self.assertNotIn("metrics", report["r2"])

    def test_preflight_failure_stops_before_any_training(self):
        runner = self.runner()
        (self.root/"tools").mkdir()
        (self.root/"tools/preflight_structure_v1.py").write_text("raise SystemExit(8)", encoding="utf-8")
        self.assertNotEqual(runner.main(self.arguments("--preflight")), 0)
        self.assertEqual(self.events(), [])

    def test_bom_in_test_list_is_rejected_before_training(self):
        runner = self.runner()
        path = self.root/"test list.txt"
        path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8-sig")
        self.assertNotEqual(runner.main(self.arguments("--runs", "r2")), 0)
        self.assertEqual(self.events(), [])
        self.assertFalse(self.output.exists())

    def test_cached_checkpoint_restores_matching_json_and_per_image_csv(self):
        runner = self.runner()
        self.assertEqual(runner.main(self.arguments("--runs", "r2")), 0)
        directory = self.output/"r2"
        best_csv = (directory/"eval_test_per_image.csv").read_bytes()
        self.assertEqual(runner.main(self.arguments("--runs", "r2", "--eval-only", "--checkpoint", "last.pt")), 0)
        self.assertNotEqual(best_csv, (directory/"eval_test_per_image.csv").read_bytes())
        before = self.events()
        self.assertEqual(runner.main(self.arguments("--runs", "r2", "--resume")), 0)
        self.assertEqual(before, self.events())
        self.assertEqual(json.loads((directory/"eval_test.json").read_text())["psnr"], 21.5)
        self.assertEqual((directory/"eval_test_per_image.csv").read_bytes(), best_csv)
        self.assertEqual((directory/"eval_test_best_per_image.csv").read_bytes(), best_csv)
        self.assertIn("last.pt", (directory/"eval_test_last_per_image.csv").read_text())

    def test_help_works_outside_project(self):
        self.assertTrue(SCRIPT.is_file(), "The V1 runner is not implemented")
        proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=self.root,
                              capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--resume", proc.stdout)

    def test_actual_cpu_train_and_evaluate_pipeline(self):
        import numpy as np
        from PIL import Image
        generator = np.random.default_rng(41)
        for path in self.root.glob("*.png"):
            Image.fromarray(generator.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
        cfg_path = self.config_dir/"structure_v1_r2.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg["data"].update(image_size=32, resize_short_to=32)
        cfg["model"].update(readout_iterations=40, readout_backward_iterations=96)
        cfg["optim"].update(warmup_steps=0, amp=False)
        cfg["structure_training"].update(warmup_epochs=0, ramp_epochs=0)
        cfg["diagnostics"].update(fixed_samples=1)
        cfg["eval"].update(compute_lpips=False, compute_fid=False)
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        runner = self.runner()
        runner.ROOT = ROOT  # Use the real train.py/evaluate.py in this one integration check.
        self.assertEqual(runner.main(self.arguments("--runs", "r2", "--epochs", "1",
            "--batch-size", "2", "--num-workers", "0")), 0)
        report = json.loads((self.output/"summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["r2"]["status"], "complete")
        self.assertGreater(report["r2"]["metrics"]["stage1_p_injection_l1"], 0)
        self.assertEqual(report["r2"]["metrics"]["rho_scale"], 1)
        self.assertTrue((self.output/"r2/best.pt").is_file())
        self.assertTrue((self.output/"r2/eval_test_per_image.csv").is_file())


if __name__ == "__main__":
    unittest.main()
