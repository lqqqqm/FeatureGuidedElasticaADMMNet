"""Behavioral checks for the standalone training-only preflight tool."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "preflight_structure_v1.py"


class StructurePreflightTests(unittest.TestCase):
    def module(self):
        self.assertTrue(SCRIPT.is_file(), "preflight CLI implementation is missing")
        spec = importlib.util.spec_from_file_location("structure_preflight", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_rho_activation_threshold_accounts_for_shrinkage_and_opposing_q(self):
        module = self.module()
        g = torch.zeros(1, 6, 1, 1)
        g[:, :3] = 0.2
        c = torch.full((1, 3, 1, 1), 1.1)
        q = torch.zeros_like(g)
        critical = module.prior_activation_threshold(q, c, g, r2=2)
        torch.testing.assert_close(critical, torch.full_like(c, 5.5))
        q[:, :3] = -0.1
        critical = module.prior_activation_threshold(q, c, g, r2=2)
        torch.testing.assert_close(critical, torch.full_like(c, 6.5))

    def test_missing_train_data_does_not_fall_back_to_test_or_synthetic(self):
        module = self.module()
        cfg = {"data": {"test_list": "do-not-open-test.txt", "image_size": 32}}
        with self.assertRaisesRegex(ValueError, "train_list"):
            module.load_samples(cfg, 1, seed=4)

    def test_synthetic_requires_explicit_flag_and_produces_reproducible_masks(self):
        module = self.module()
        cfg = {"data": {"image_size": 256}}
        first = module.load_samples(cfg, 2, seed=5, synthetic=True, synthetic_size=32)
        second = module.load_samples(cfg, 2, seed=5, synthetic=True, synthetic_size=32)
        for a, b in zip(first, second):
            torch.testing.assert_close(a["gt"], b["gt"])
            torch.testing.assert_close(a["mask"], b["mask"])
            self.assertGreater(a["mask"].sum().item(), 0)
            self.assertGreater((1 - a["mask"]).sum().item(), 0)
            self.assertEqual(a["gt"].shape, (1, 3, 32, 32))
        self.assertEqual(cfg["data"]["image_size"], 256)

    def test_backward_sweep_uses_normalized_hole_loss_and_relative_adjoint_error(self):
        module = self.module()
        self.assertTrue(hasattr(module, "backward_sweep"))
        torch.set_num_threads(1)
        gt = torch.zeros(1, 3, 16, 16)
        pred = torch.ones_like(gt)
        mask = torch.ones(1, 1, 16, 16)
        mask[..., 2:14, 2:14] = 0
        result = module.backward_sweep(pred, gt, mask, 2., 10., 1024, 1e-7)
        self.assertEqual(result["reference"].get("reference_dtype"), "float64")
        self.assertLess(result["reference"]["relative_residual_max"], 1e-8)
        self.assertAlmostEqual(result["upstream_gradient_l1"], 1, places=6)
        self.assertFalse(result["gradient_is_zero"])
        self.assertEqual([row["iterations"] for row in result["budgets"]], [160, 320, 640])
        for row in result["budgets"]:
            self.assertLess(row["relative_residual_max"], 1e-4)
            self.assertLess(row["hole_adjoint_relative_l1_vs_reference"], 1e-3)
            self.assertLess(row["adjoint_relative_l2_vs_reference"], 1e-3)
            self.assertLess(row["gradient_relative_l2_vs_reference"], 1e-3)
            self.assertLess(row["hole_gradient_relative_l2_vs_reference"], 1e-3)

    def test_backward_sweep_zero_loss_gradient_is_explicit_and_finite(self):
        module = self.module()
        self.assertTrue(hasattr(module, "backward_sweep"))
        gt = torch.zeros(1, 3, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        mask[..., 2:6, 2:6] = 0
        result = module.backward_sweep(gt, gt, mask, 2., 10., 1024, 1e-7)
        self.assertTrue(result["gradient_is_zero"])
        for row in result["budgets"]:
            self.assertEqual(row["relative_residual_max"], 0)
            self.assertEqual(row["hole_adjoint_relative_l1_vs_reference"], 0)
            self.assertIn("gradient_relative_l2_vs_reference", row)
            self.assertEqual(row["gradient_relative_l2_vs_reference"], 0)

    def test_reference_budget_must_exceed_largest_forward_budget(self):
        with tempfile.TemporaryDirectory(prefix="structure-preflight-budget-test-") as directory:
            command = [sys.executable, str(SCRIPT), "--config", str(ROOT / "configs" / "mvp.yaml"),
                       "--samples", "1", "--device", "cpu", "--synthetic",
                       "--reference-iterations", "512", "--output", str(Path(directory) / "report.json")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 2, "An inadequate reference budget was accepted")
            self.assertIn("512", result.stderr)

    def test_cli_runs_from_another_directory_and_reports_actual_readout_residuals(self):
        self.assertTrue(SCRIPT.is_file(), "preflight CLI implementation is missing")
        with tempfile.TemporaryDirectory(prefix="structure-preflight-test-") as directory:
            output = Path(directory) / "report.json"
            command = [sys.executable, str(SCRIPT), "--config", str(ROOT / "configs" / "mvp.yaml"),
                       "--samples", "1", "--device", "cpu", "--synthetic", "--synthetic-size", "32",
                       "--output", str(output)]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["data_source"], "synthetic_diagnostic_only")
            self.assertTrue(report["diagnostic_model"]["added_prior_for_oracle"])
            self.assertEqual(report["weights"], "random_initialization")
            self.assertEqual(report.get("reference_dtype"), "float64")
            self.assertGreater(report["recommendations"]["rho_reference"], 0)
            sample = report["samples"][0]
            for scenario in ("injection_disabled", "oracle_gt_prior"):
                sweep = sample[scenario]["readout_sweep"]
                self.assertEqual([row["iterations"] for row in sweep], [20, 40, 80, 160, 320, 512])
                for row in sweep:
                    self.assertGreaterEqual(row["relative_residual_max"], 0)
                    self.assertGreaterEqual(row["hole_l1_vs_reference"], 0)
                self.assertEqual(sample[scenario]["reference"]["iterations"], 1024)
                self.assertEqual(sample[scenario]["reference"].get("reference_dtype"), "float64")
                backward = sample[scenario]["backward_sweep"]
                self.assertEqual([row["iterations"] for row in backward["budgets"]], [160, 320, 640])
                self.assertEqual(backward["reference"]["iterations"], 1024)
                self.assertEqual(backward["reference"].get("reference_dtype"), "float64")
            self.assertIn("backward_iterations_reference", report["recommendations"])
            self.assertIn("oracle_minus_disabled_hole_l1", sample)


if __name__ == "__main__":
    unittest.main()
