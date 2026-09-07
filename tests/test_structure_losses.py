"""CPU-only behavioral checks for structure targets and supervision."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
import unittest

import torch

from fg_elastica_inpaint.losses.inpainting import InpaintingLoss
from fg_elastica_inpaint.models.operators import grad


class StructureLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def helpers(self):
        name = "fg_elastica_inpaint.utils.structure"
        self.assertIsNotNone(importlib.util.find_spec(name), "shared structure targets are missing")
        return importlib.import_module(name)

    def criterion(self, **kwargs):
        self.assertIn("lambda_structure_grad", inspect.signature(InpaintingLoss).parameters)
        options = dict(K=1, lambda_rec=0, lambda_edge=0, lambda_stage=0,
                       structure_known_weight=0)
        options.update(kwargs)
        return InpaintingLoss(**options)

    @staticmethod
    def outputs(gt, gradients, logits=None):
        if logits is None:
            logits = torch.zeros_like(gt[:, :1], requires_grad=True)
        return {"pred": gt.clone().requires_grad_(), "stage_preds": [gt], "aux": {},
                "structure": {"gradient": gradients[0], "gradient_pyramid": gradients,
                              "edge_logits": logits, "edge": logits.sigmoid()}}

    @staticmethod
    def ramp(size=8):
        return torch.arange(size, dtype=torch.float32).view(1, 1, 1, size).expand(1, 3, size, size) / size

    def test_legacy_edge_target_is_zero_for_constant_image(self):
        target = InpaintingLoss._edge_target(torch.full((2, 3, 8, 8), 0.4))
        self.assertEqual(torch.count_nonzero(target).item(), 0)

    def test_shared_edge_strength_uses_vector_norm_and_fixed_scale(self):
        helper = self.helpers()
        g = torch.tensor([3., 0., 0., 4., 0., 0.]).view(1, 6, 1, 1)
        self.assertAlmostEqual(helper.edge_strength(g, scale=1).item(), 1 - math.exp(-5 / 3), places=6)
        zero = torch.zeros(2, 6, 4, 4, requires_grad=True)
        edge = helper.edge_strength(zero)
        self.assertEqual(torch.count_nonzero(edge).item(), 0)
        edge.sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())
        with self.assertRaises(ValueError):
            helper.edge_strength(g, scale=0)

    def test_multiscale_targets_differentiate_antialiased_resized_image(self):
        helper = self.helpers()
        gt = self.ramp() * 8
        targets = helper.structure_targets(gt, [(8, 8), (4, 4), (2, 2)])
        half = targets["gradient_pyramid"][1]
        expected_dx = torch.tensor([25 / 14, 2, 25 / 14, 0])
        torch.testing.assert_close(half[0, 0, 1], expected_dx, rtol=1e-5, atol=1e-6)
        self.assertEqual(torch.count_nonzero(half[:, 3:]).item(), 0)
        constant = helper.structure_targets(torch.ones_like(gt), [(8, 8), (4, 4), (2, 2)])
        for edge in constant["edge_pyramid"]:
            self.assertEqual(torch.count_nonzero(edge).item(), 0)

    def test_multiscale_gradient_supervision_rejects_zero_and_flipped_vectors(self):
        helper = self.helpers()
        gt = self.ramp()
        gradients = helper.structure_targets(gt, [(8, 8), (4, 4), (2, 2)])["gradient_pyramid"]
        criterion = self.criterion(lambda_structure_grad=1)
        mask = torch.zeros_like(gt[:, :1])
        matched = criterion(self.outputs(gt, gradients), gt, mask)
        self.assertAlmostEqual(matched["structure_grad"].item(), 0, places=7)
        for scale in range(3):
            corrupted = list(gradients)
            corrupted[scale] = torch.zeros_like(corrupted[scale])
            self.assertGreater(criterion(self.outputs(gt, corrupted), gt, mask)["total"].item(), 0)
        flipped = criterion(self.outputs(gt, [-g for g in gradients]), gt, mask)
        self.assertGreater(flipped["structure_grad"].item(), 0)

    def test_gradient_loss_normalizes_hole_size(self):
        gt = torch.zeros(2, 3, 8, 8)
        prediction = torch.full((2, 6, 8, 8), 0.2, requires_grad=True)
        mask = torch.ones(2, 1, 8, 8)
        mask[0, :, :1] = 0
        mask[1, :, :6] = 0
        result = self.criterion(lambda_structure_grad=1)(self.outputs(gt, [prediction]), gt, mask)
        self.assertAlmostEqual(result["structure_grad"].item(), 0.2, places=6)

    def test_sparse_edges_and_non_edges_receive_balanced_gradient_supervision(self):
        gt = torch.zeros(1, 3, 8, 32)
        gt[..., 16:] = 1
        target = grad(gt)
        edge_error = torch.zeros_like(target)
        edge_error[..., 15] = 0.2
        flat_error = torch.full_like(target, 0.2)
        flat_error[..., 15] = 0
        criterion = self.criterion(lambda_structure_grad=1)
        mask = torch.zeros_like(gt[:, :1])
        edge_loss = criterion(self.outputs(gt, [target + edge_error]), gt, mask)["structure_grad"]
        flat_loss = criterion(self.outputs(gt, [target + flat_error]), gt, mask)["structure_grad"]
        self.assertAlmostEqual(edge_loss.item(), flat_loss.item(), delta=2e-5)

    def test_known_gradient_supervision_is_separate_and_weaker(self):
        gt = torch.zeros(1, 3, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 4:] = 1
        criterion = self.criterion(lambda_structure_grad=1, structure_known_weight=0.1)
        known = mask.expand(1, 6, 8, 8) * 0.2
        hole = (1 - mask).expand(1, 6, 8, 8) * 0.2
        known_loss = criterion(self.outputs(gt, [known]), gt, mask)["total"]
        hole_loss = criterion(self.outputs(gt, [hole]), gt, mask)["total"]
        self.assertAlmostEqual(known_loss.item() / hole_loss.item(), 0.1, places=6)

    def test_orientation_is_signed_and_uses_gt_edge_support(self):
        gt = self.ramp()
        target = grad(gt)
        mask = torch.zeros_like(gt[:, :1])
        criterion = self.criterion(lambda_structure_orientation=1)
        for vector, expected in ((target, 0), (-target, 2), (torch.zeros_like(target), 1)):
            loss = criterion(self.outputs(gt, [vector]), gt, mask)["structure_orientation"]
            self.assertAlmostEqual(loss.item(), expected, places=5)
        constant = torch.zeros_like(gt)
        loss = criterion(self.outputs(constant, [target]), constant, mask)["structure_orientation"]
        self.assertEqual(loss.item(), 0)

    def test_empty_holes_and_edge_sets_have_finite_zero_losses_and_backward(self):
        gt = torch.zeros(1, 3, 8, 8)
        vector = torch.zeros(1, 6, 8, 8, requires_grad=True)
        outputs = self.outputs(gt, [vector])
        criterion = self.criterion(lambda_structure_grad=1, lambda_structure_edge=1,
                                   lambda_structure_orientation=1, lambda_structure_consistency=1)
        result = criterion(outputs, gt, torch.ones_like(gt[:, :1]))
        self.assertEqual(result["total"].item(), 0)
        result["total"].backward()
        self.assertTrue(torch.isfinite(vector.grad).all())
        self.assertTrue(torch.isfinite(outputs["structure"]["edge_logits"].grad).all())

    def test_predicted_edge_cannot_turn_off_gradient_supervision(self):
        gt = self.ramp()
        g = torch.zeros(1, 6, 8, 8)
        criterion = self.criterion(lambda_structure_grad=1)
        losses = []
        for value in (-20., 20.):
            logits = torch.full_like(gt[:, :1], value)
            losses.append(criterion(self.outputs(gt, [g], logits), gt, torch.zeros_like(logits))["total"])
        torch.testing.assert_close(losses[0], losses[1])
        self.assertGreater(losses[0].item(), 0)

    def test_edge_logits_receive_supervision_on_constant_and_strong_edge_gt(self):
        criterion = self.criterion(lambda_structure_edge=1)
        for strong in (False, True):
            gt = torch.zeros(1, 3, 8, 8)
            if strong:
                gt[..., 4:] = 1
            mask = torch.ones_like(gt[:, :1])
            mask[..., 3] = 0
            outputs = self.outputs(gt, [grad(gt)])
            result = criterion(outputs, gt, mask)
            result["total"].backward()
            derivative = outputs["structure"]["edge_logits"].grad[..., 3].mean().item()
            self.assertLess(derivative, 0) if strong else self.assertGreater(derivative, 0)

    def test_consistency_backpropagates_to_both_gradient_and_edge_heads(self):
        gt = torch.zeros(1, 3, 8, 8)
        vector = torch.full((1, 6, 8, 8), 0.02, requires_grad=True)
        outputs = self.outputs(gt, [vector])
        criterion = self.criterion(lambda_structure_consistency=1)
        result = criterion(outputs, gt, torch.zeros_like(gt[:, :1]))
        result["total"].backward()
        self.assertGreater(vector.grad.abs().sum().item(), 0)
        self.assertGreater(outputs["structure"]["edge_logits"].grad.abs().sum().item(), 0)

    def test_config_routes_new_weights_and_old_defaults_disable_structure_losses(self):
        self.assertTrue(hasattr(InpaintingLoss, "from_config"))
        gt = self.ramp()
        outputs = self.outputs(gt, [torch.zeros(1, 6, 8, 8)])
        mask = torch.zeros_like(gt[:, :1])
        old = InpaintingLoss.from_config({"model": {"K": 1}, "loss": {}})(outputs, gt, mask)
        for name in ("structure_grad", "structure_edge", "structure_orientation", "structure_consistency"):
            self.assertEqual(old[name].item(), 0)
        config = {"model": {"K": 1}, "loss": {"lambda_rec": 0, "lambda_edge": 0,
                  "lambda_stage": 0, "lambda_structure_grad": 2.5, "structure_known_weight": 0,
                  "structure_scale_weights": [1], "edge_scale": 0.2}}
        result = InpaintingLoss.from_config(config)(outputs, gt, mask)
        self.assertAlmostEqual(result["total"].item(), 2.5 * result["structure_grad"].item(), places=6)

    def test_readout_reconstruction_is_logged_without_double_counting(self):
        gt = torch.zeros(1, 3, 8, 8)
        outputs = {"pred": torch.ones_like(gt, requires_grad=True), "stage_preds": [gt],
                   "structure": None, "readout": {"relative_residual": torch.tensor(0.)}}
        criterion = self.criterion(lambda_rec=1)
        result = criterion(outputs, gt, torch.zeros_like(gt[:, :1]))
        self.assertAlmostEqual(result["total"].item(), 6)
        self.assertAlmostEqual(result["readout"].item(), 6)
        self.assertFalse(result["readout"].requires_grad)
        for name in ("rec", "edge", "perc", "stage", "p_cons", "n_m", "struct",
                     "structure_grad", "structure_edge", "structure_orientation", "structure_consistency"):
            self.assertIn(name, result)


if __name__ == "__main__":
    unittest.main()
