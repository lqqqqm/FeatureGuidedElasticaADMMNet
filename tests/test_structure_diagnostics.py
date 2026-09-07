"""Small CPU inference checks for real-forward trace and interventions."""

import importlib
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

from fg_elastica_inpaint.models import FeatureGuidedElasticaADMMNet
from fg_elastica_inpaint.models.unfolding import StageHyperParams
import trace_infer


class StructureDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(571)
        cls.model = FeatureGuidedElasticaADMMNet(
            image_size=32, K=1, transformer_depth=0, use_structure_prior=True,
            structure_rho=2., use_pcg_readout=True, readout_iterations=25,
            stage_hyper=StageHyperParams(Tu=1, Tn=1),
        ).eval()
        cls.gt = torch.rand(1, 3, 32, 32) * 2 - 1
        cls.mask = torch.ones(1, 1, 32, 32)
        cls.mask[..., 10:22, 10:22] = 0

    def test_trace_contains_exact_actual_readout_and_stage_tensors(self):
        with torch.no_grad():
            expected = self.model(self.gt * self.mask, self.mask, return_stage_states=True)
        trace, stats = trace_infer.trace_forward(self.model, self.gt, self.mask)
        self.assertTrue(torch.equal(trace["pred"], expected["pred"]))
        self.assertTrue(torch.equal(trace["comp"], expected["comp"]))
        self.assertTrue(torch.equal(trace["stage_1.p"], expected["stage_states"][0]["p"]))
        self.assertTrue(torch.equal(trace["readout.relative_residual"], expected["readout"]["relative_residual"]))
        self.assertTrue(torch.equal(trace["structure.gradient"], expected["structure"]["gradient"]))
        self.assertIn("diagnostics.stage_1.p_injection_l1", trace)
        self.assertTrue(json.dumps(stats, allow_nan=False))

    def test_gray_fixed_scale_retains_absolute_edge_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edge.png"
            values = torch.tensor([[[[0., .25], [.5, 1.]]]])
            self.assertIn("value_range", __import__("inspect").signature(trace_infer.save_gray_map).parameters)
            trace_infer.save_gray_map(values, path, value_range=(0., 1.))
            self.assertEqual(np.asarray(Image.open(path)).tolist(), [[0, 64], [128, 255]])

    def test_interventions_are_repeatable_and_leave_model_unchanged(self):
        self.assertIsNotNone(importlib.util.find_spec("tools.diagnose_structure_prior"))
        diagnostic = importlib.import_module("tools.diagnose_structure_prior")
        with torch.no_grad():
            before = self.model(self.gt * self.mask, self.mask)["pred"].clone()
        with tempfile.TemporaryDirectory() as directory:
            report = diagnostic.run_diagnostics(self.model, self.gt, self.mask, Path(directory), seed=19)
            self.assertEqual(set(report["modes"]), {
                "predicted", "disabled_prior", "spatial_shuffled", "oracle_gt_gradient", "disable_correction",
            })
            self.assertTrue(report["oracle_is_diagnostic_only"])
            for mode in report["modes"]:
                self.assertTrue((Path(directory) / mode / "raw_tensors.pt").is_file())
                self.assertTrue((Path(directory) / mode / "images" / "comp.png").is_file())
                self.assertTrue((Path(directory) / mode / "structure" / "edge.png").is_file())
            self.assertGreater(report["modes"]["disabled_prior"]["output_change"]["hole_l1"], 0)
            oracle = torch.load(Path(directory) / "oracle_gt_gradient" / "raw_tensors.pt", weights_only=True)
            self.assertTrue(torch.equal(oracle["prior.gradient"], oracle["target.gradient"]))
            self.assertTrue(json.dumps(report, allow_nan=False))
        with torch.no_grad():
            after = self.model(self.gt * self.mask, self.mask)["pred"]
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(self.model.structure_rho, 2.)
        gradient = torch.arange(6 * 16).reshape(1, 6, 4, 4).float()
        shuffled = diagnostic.spatial_shuffle_gradient(gradient, seed=19)
        self.assertTrue(torch.equal(shuffled, diagnostic.spatial_shuffle_gradient(gradient, seed=19)))
        self.assertFalse(torch.equal(shuffled, gradient))
        self.assertTrue(torch.equal(torch.sort(shuffled.flatten(2), dim=-1).values,
                                    torch.sort(gradient.flatten(2), dim=-1).values))
        # The same spatial permutation retains each six-dimensional vector.
        self.assertTrue(torch.all(shuffled[:, 1] - shuffled[:, 0] == 16))

    def test_both_cli_entrypoints_load_one_temporary_checkpoint(self):
        cfg = {
            "data": {"image_size": 32},
            "model": {"K": 1, "transformer_depth": 0, "use_structure_prior": True,
                      "structure_rho": 2., "use_pcg_readout": True, "readout_iterations": 25},
            "stage_hyper": {"Tu": 1, "Tn": 1},
        }
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = temporary / "config.yaml"
            config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            checkpoint = temporary / "temporary_random_weights.pt"
            torch.save({"model": self.model.state_dict()}, checkpoint)
            image_path, mask_path = temporary / "image.png", temporary / "mask.png"
            trace_infer.save_rgb(self.gt, image_path)
            trace_infer.save_gray_map(self.mask, mask_path, value_range=(0., 1.))
            for script, output_name in ((root / "trace_infer.py", "trace"),
                                        (root / "tools" / "diagnose_structure_prior.py", "diagnosis")):
                output = temporary / output_name
                result = subprocess.run([
                    sys.executable, str(script), "--config", str(config), "--checkpoint", str(checkpoint),
                    "--image", str(image_path), "--mask", str(mask_path), "--output_dir", str(output),
                    "--device", "cpu",
                ], cwd=root, text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            traced = torch.load(temporary / "trace" / "raw_tensors.pt", weights_only=True)
            predicted = torch.load(temporary / "diagnosis" / "predicted" / "raw_tensors.pt", weights_only=True)
            self.assertTrue(torch.equal(traced["pred"], predicted["pred"]))
            self.assertTrue((temporary / "diagnosis" / "diagnostics.json").is_file())
        self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
