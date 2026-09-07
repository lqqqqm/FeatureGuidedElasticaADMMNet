import csv
import json
from pathlib import Path
import random
import tempfile
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image
import torch

from fg_elastica_inpaint.data.dataset import InpaintingImageDataset
from fg_elastica_inpaint.losses import InpaintingLoss
from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.utils.config import load_config
from fg_elastica_inpaint.utils.diagnostics import coupling_scale
from fg_elastica_inpaint.utils.misc import set_seed
from train import train_one_epoch, validate


ROOT = Path(__file__).resolve().parents[1]


class StructureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        cfg = load_config(ROOT / "configs/structure_v1_r2.yaml")
        cfg["data"]["image_size"] = 32
        cfg["model"].update(readout_iterations=40, readout_backward_iterations=96)
        cfg["eval"].update(compute_lpips=False, compute_fid=False)
        cfg["diagnostics"].update(fixed_samples=2, raw_every=1)
        cfg["structure_training"].update(warmup_epochs=0, ramp_epochs=0)
        return cfg

    def test_matched_r0_r1_initialization_and_zero_coupling(self):
        cfg = self.config()
        cfg["model"].update(use_structure_prior=False, structure_rho=0)
        set_seed(42)
        baseline = FeatureGuidedElasticaADMMNet.from_config(cfg).eval()
        cfg["model"]["use_structure_prior"] = True
        set_seed(42)
        head = FeatureGuidedElasticaADMMNet.from_config(cfg).eval()
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, head.state_dict()[key]), key)
        x = torch.rand(1, 3, 32, 32)*2-1
        M = torch.ones_like(x[:, :1]); M[..., 7:26, 5:27] = 0
        with torch.no_grad():
            a, b = baseline(x*M, M), head(x*M, M)
        self.assertTrue(torch.equal(a["pred"], b["pred"]))
        self.assertTrue(torch.equal(a["aux"]["p"], b["aux"]["p"]))

    def test_three_training_steps_and_validation_keep_structure_evidence(self):
        cfg = self.config()
        set_seed(17)
        model = FeatureGuidedElasticaADMMNet.from_config(cfg)
        criterion = InpaintingLoss.from_config(cfg)
        gt = torch.rand(2, 3, 32, 32)*2-1
        M = torch.ones_like(gt[:, :1]); M[..., 5:28, 8:28] = 0
        batch = dict(gt=gt, mask=M, masked=gt*M, path=["synthetic0", "synthetic1"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        before = model.structure_prior.gradient1.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stats = train_one_epoch(model, [batch]*3, optimizer, scheduler, criterion,
                                    scaler, torch.device("cpu"), cfg, 1, out)
            self.assertTrue(all(np.isfinite(v) for v in stats.values() if isinstance(v, (int, float))))
            self.assertGreater(stats["structure_grad_norm"], 0)
            self.assertGreater(stats["stage3_p_injection_l1"], 0)
            self.assertFalse(torch.equal(before, model.structure_prior.gradient1.weight))
            metrics = validate(model, [batch], criterion, torch.device("cpu"), cfg, 1, out)
            for key in ("hole_psnr", "hole_ssim", "hole_edge_f1", "structure_gradient_l1", "readout_relative_residual"):
                self.assertTrue(np.isfinite(metrics[key]), key)
            with (out/"per_image/val_0001.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertIn("mask_sha256", rows[0])
            self.assertTrue((out/"diagnostics/val_0001/sample_001/raw_tensors.pt").is_file())
            with torch.no_grad():
                active = model(gt*M, M)
                disabled = model(gt*M, M, rho_scale=0)
            self.assertTrue(torch.isfinite(active["pred"]).all())
            self.assertGreater(float((active["pred"]-disabled["pred"]).abs().max()), 1e-6)
            # Isolate the RGB -> readout -> p -> G path from all explicit prior losses.
            active = model(gt*M, M)
            gradient, = torch.autograd.grad(active["pred"].square().mean(), active["structure"]["gradient"])
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_standalone_evaluation_preserves_checkpoint_coupling_and_diagnostics(self):
        from fg_elastica_inpaint.utils.config import save_config
        cfg = self.config()
        cfg["structure_training"] = {"warmup_epochs": 2, "ramp_epochs": 3}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image = directory/"sample.png"
            pixels = np.random.default_rng(9).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(image)
            image_list = directory/"images.txt"
            image_list.write_text(str(image), encoding="utf-8")
            cfg["data"].update(train_list=str(image_list), val_list=str(image_list), test_list=None,
                               num_workers=0, resize_short_to=32, val_batch_size=1)
            config = directory/"config.yaml"
            save_config(cfg, config)
            model = FeatureGuidedElasticaADMMNet.from_config(cfg)
            checkpoint = directory/"weights.pt"
            torch.save({"model": model.state_dict(), "epoch": 1, "structure_rho_scale": 0.}, checkpoint)
            completed = subprocess.run([sys.executable, str(ROOT/"evaluate.py"), "--config", str(config),
                "--checkpoint", str(checkpoint), "--split", "val"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            result = json.loads((directory/"eval_val.json").read_text(encoding="utf-8"))
            self.assertEqual(result["rho_scale"], 0.)
            self.assertEqual(result["stage1_p_injection_l1"], 0.)
            self.assertIn("readout_relative_residual", result)
            self.assertIn("hole_psnr", result)

    def test_eval_masks_are_fixed_and_do_not_change_training_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"image.png"
            Image.new("RGB", (32, 32), "gray").save(p)
            ds = InpaintingImageDataset([str(p)], image_size=32, resize_short_to=32, train=False)
            set_seed(5)
            before = random.getstate()
            mask = ds[0]["mask"]
            self.assertEqual(before, random.getstate())
            set_seed(999)
            self.assertTrue(torch.equal(mask, ds[0]["mask"]))
        cfg = {"structure_training": {"warmup_epochs": 2, "ramp_epochs": 3}}
        self.assertEqual([coupling_scale(cfg, x) for x in (0, 2, 3.5, 5, 6)], [0, 0, .5, 1, 1])


if __name__ == "__main__":
    unittest.main()
