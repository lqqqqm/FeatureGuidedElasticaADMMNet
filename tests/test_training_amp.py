"""Exercise real GradScaler overflow handling on CPU without a CUDA dependency."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from train import train_one_epoch


class HalfBoundaryModel(torch.nn.Module):
    """A finite forward whose scaled VJP overflows at the FP16 boundary."""

    def __init__(self, invalid_backward=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.structure_prior = None
        self.invalid_backward = invalid_backward
        self.weights_seen = []

    def forward(self, image, mask, **kwargs):
        self.weights_seen.append(float(self.weight.detach()))
        if self.invalid_backward:
            # sqrt(0) has a finite value but an infinite derivative at every scale.
            value = (self.weight - 1).sqrt()
        else:
            value = self.weight.half().float()
        return {"pred": image * value, "diagnostics": []}


def criterion(outputs, gt, mask):
    return {"total": outputs["pred"].mean()}


class TrainingAmpTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        value = torch.ones(1, 1, 1, 1)
        self.batch = {"gt": value, "mask": value, "masked": value, "path": ["synthetic.png"]}
        self.cfg = {"optim": {"amp": True, "grad_clip": 1.}, "structure_training": {}}

    def run_epoch(self, model, batches=2, enabled=True):
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        scaler = torch.amp.GradScaler("cpu", enabled=enabled)
        stats = train_one_epoch(model, [self.batch]*batches, optimizer, scheduler, criterion,
                                scaler, torch.device("cpu"), self.cfg, 1, self.output)
        return stats, optimizer, scheduler, scaler

    def test_overflow_backs_off_without_updating_weights_or_scheduler(self):
        model = HalfBoundaryModel()
        stats, optimizer, scheduler, scaler = self.run_epoch(model)
        self.assertEqual(model.weights_seen, [1., 1.])
        self.assertLess(float(model.weight.detach()), 1.)
        self.assertEqual(float(optimizer.state[model.weight]["step"]), 1.)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(scaler.get_scale(), 32768.)
        self.assertEqual(stats["optimizer_steps"], 1)
        self.assertEqual(stats["amp_skipped_steps"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(stats["grad_norm"])))
        with (self.output/"amp_events.csv").open(encoding="utf-8") as stream:
            events = list(csv.DictReader(stream))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["bad_parameters"], "weight")
        self.assertEqual(float(events[0]["scale_after"]), 32768.)

    def test_persistent_nonfinite_backward_stops_with_parameter_diagnostics(self):
        model = HalfBoundaryModel(invalid_backward=True)
        with self.assertRaisesRegex(FloatingPointError, "Persistent.*weight"):
            self.run_epoch(model, batches=20)
        self.assertEqual(float(model.weight.detach()), 1.)
        self.assertLess(len(model.weights_seen), 20)

    def test_nonfinite_backward_without_scaling_still_fails_immediately(self):
        model = HalfBoundaryModel(invalid_backward=True)
        with self.assertRaisesRegex(FloatingPointError, "weight"):
            self.run_epoch(model, enabled=False)
        self.assertEqual(len(model.weights_seen), 1)
        self.assertEqual(float(model.weight.detach()), 1.)

    def test_epoch_without_any_update_is_not_reported_as_success(self):
        with self.assertRaisesRegex(FloatingPointError, "No optimizer updates"):
            self.run_epoch(HalfBoundaryModel(), batches=1)

    def test_unrelated_clipping_errors_are_not_classified_as_amp_overflow(self):
        for error in (RuntimeError("unrelated clipping failure"), torch.OutOfMemoryError("allocation failed")):
            with self.subTest(error=type(error).__name__):
                with patch("train.torch.nn.utils.clip_grad_norm_", side_effect=error):
                    with self.assertRaisesRegex(type(error), str(error)):
                        self.run_epoch(HalfBoundaryModel(), batches=1)
                self.assertFalse((self.output/"amp_events.csv").exists())


if __name__ == "__main__":
    unittest.main()
