"""Small checks for the cross-version evaluation contract (no GPU/downloads)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/compare_v0_v1.py"


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="model comparison ")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def module(self):
        self.assertTrue(SCRIPT.exists(), "The unified comparison entry point is missing")
        spec = importlib.util.spec_from_file_location("comparison_test_module", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_list_identity_handles_windows_separators_and_case(self):
        m = self.module()
        self.assertEqual(m.path_key(r"C:\dataset\A.jpg"), m.path_key("c:/dataset/a.jpg"))

    def test_protocol_rejects_reusing_different_checkpoint(self):
        m = self.module()
        out = self.base / "run"
        m.accept_protocol(out, {"weights": "first"}, resume=False)
        with self.assertRaisesRegex(ValueError, "protocol|settings"):
            m.accept_protocol(out, {"weights": "second"}, resume=True)
        self.assertEqual(json.loads((out / "protocol.json").read_text())["weights"], "first")

    def test_prediction_integrity_detects_corruption(self):
        m = self.module()
        path = self.base / "000000.npz"
        pred = np.zeros((3, 8, 8), np.float32)
        m.save_prediction(path, pred, "input-a")
        self.assertTrue(m.valid_prediction(path, "input-a", 8))
        self.assertFalse(m.valid_prediction(path, "input-b", 8))
        path.write_bytes(b"interrupted write")
        self.assertFalse(m.valid_prediction(path, "input-a", 8))

    def test_metric_comparison_direction_and_missing_values(self):
        m = self.module()
        rows = m.metric_deltas({"hole_psnr": 20., "lpips": .2},
                               {"hole_psnr": 21., "lpips": .15})
        values = {r["metric"]: r for r in rows}
        self.assertAlmostEqual(values["hole_psnr"]["delta_v1_minus_v0"], 1.)
        self.assertTrue(values["lpips"]["improved"])
        self.assertAlmostEqual(values["lpips"]["relative_improvement_percent"], 25.)
        self.assertIsNone(values["hole_psnr"]["relative_improvement_percent"])

    def test_training_overlap_is_rejected(self):
        m = self.module()
        image = self.base / "a.jpg"
        image.write_bytes(b"not opened for list audit")
        train = self.base / "train.txt"
        val = self.base / "val.txt"
        train.write_text(str(image), encoding="utf-8")
        val.write_text(str(image), encoding="utf-8")
        cfg = {"data": {"train_list": str(train), "val_list": str(val)}}
        with self.assertRaisesRegex(ValueError, "training|overlap"):
            m.audit_lists(cfg, cfg, None)

    def test_end_to_end_isolates_same_package_name_and_resumes(self):
        self.module()
        train = self.base / "train.txt"
        val = self.base / "val.txt"
        train.write_text(str(self.base / "unseen_training.jpg"), encoding="utf-8")
        paths = []
        for i in range(2):
            path = self.base / f"val_{i}.png"
            arr = np.zeros((32, 32, 3), dtype=np.uint8)
            arr[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 7
            arr[:, :, 1] = 100 + i * 30
            Image.fromarray(arr).save(path)
            paths.append(str(path))
        val.write_text("\n".join(paths), encoding="utf-8")
        model_source = '''
import torch
class FeatureGuidedElasticaADMMNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
    @classmethod
    def from_config(cls, cfg): return cls()
    def forward(self, image, mask%s):
        pred = image + (1-mask)*(self.bias+%s)
        return {"pred":pred,"comp":mask*image+(1-mask)*pred}
'''
        args = [sys.executable, "-B", str(SCRIPT), "--device", "cpu", "--no-lpips",
                "--batch-size", "1", "--visual-count", "2", "--output-dir", str(self.base / "output")]
        for label, offset in [("old", "0.0"), ("v1", "0.2")]:
            project = self.base / label
            package = project / "fg_elastica_inpaint" / "models"
            package.mkdir(parents=True)
            (package.parent / "__init__.py").write_text("")
            (package / "__init__.py").write_text("from .network import FeatureGuidedElasticaADMMNet\n")
            (package / "network.py").write_text(model_source % (", rho_scale=1.0" if label == "v1" else "", offset))
            cfg = {"model": {}, "stage_hyper": {}, "data": {"train_list": str(train),
                   "val_list": str(val), "image_size": 32, "resize_short_to": 32,
                   "hole_buckets": [[.4,.5]], "bucket_probs": [1],
                   "mask_topology_modes": ["large_contiguous"], "mask_topology_probs": [1]}}
            config = project / "config.yaml"
            config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            ckpt = project / "best.pt"
            torch.save({"model": {"bias": torch.tensor(0.)}, "config": cfg, "epoch": 8}, ckpt)
            args += [f"--{label}-project", str(project), f"--{label}-checkpoint", str(ckpt),
                     f"--{label}-config", str(config)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = self.base / "output"
        with np.load(out / "predictions/old/000000.npz") as a, np.load(out / "predictions/v1/000000.npz") as b:
            self.assertGreater(float(np.abs(a["pred"]-b["pred"]).max()), .19)
            self.assertEqual(str(a["input_id"]), str(b["input_id"]))
        self.assertTrue((out / "report/comparison.csv").is_file())
        self.assertTrue((out / "figures/sample_000/comparison.png").is_file())
        stamp = (out / "predictions/old/000000.npz").stat().st_mtime_ns
        resumed = subprocess.run(args + ["--resume"], capture_output=True, text=True, timeout=180)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual(stamp, (out / "predictions/old/000000.npz").stat().st_mtime_ns)
        # Config must describe the trained model even if the tensors would load.
        config = self.base / "old/config.yaml"
        changed = yaml.safe_load(config.read_text())
        changed["model"] = {"K": 5}
        config.write_text(yaml.safe_dump(changed))
        bad_args = list(args)
        bad_args[bad_args.index("--output-dir")+1] = str(self.base / "mismatch")
        mismatch = subprocess.run(bad_args, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("differs from the checkpoint", mismatch.stdout + mismatch.stderr)
        self.assertFalse((self.base / "mismatch/predictions").exists())


if __name__ == "__main__":
    unittest.main()
