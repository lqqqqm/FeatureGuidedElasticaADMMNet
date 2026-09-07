"""Behavioral tests for mask-bucket and structure-evaluation contracts."""

import math
import random
import unittest

import torch

from fg_elastica_inpaint.data.mask_generator import RandomMaskGenerator
from fg_elastica_inpaint.utils import metrics


class MaskBucketTests(unittest.TestCase):
    def setUp(self):
        random.seed(173)

    def test_unreachable_drawing_distribution_still_honors_bucket(self):
        generator = RandomMaskGenerator(
            hole_buckets=[(.9, .95)], bucket_probs=[1],
            topology_modes=["scattered_irregular"], topology_probs=[1],
        )
        mask = generator(64, 64)
        ratio = (mask == 0).sum().item() / mask.numel()
        self.assertGreaterEqual(ratio, .9)
        self.assertLessEqual(ratio, .95)
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))

    def test_every_topology_supports_small_and_production_sizes(self):
        for size in (32, 64, 256):
            for topology in ("thin_distributed", "scattered_irregular", "large_contiguous", "mixed"):
                with self.subTest(size=size, topology=topology):
                    generator = RandomMaskGenerator(
                        hole_buckets=[(.20, .21)], bucket_probs=[1],
                        topology_modes=[topology], topology_probs=[1],
                    )
                    mask = generator(size, size)
                    ratio = (mask == 0).sum().item() / mask.numel()
                    self.assertGreaterEqual(ratio, .20)
                    self.assertLessEqual(ratio, .21)

    def test_bucket_with_no_representable_pixel_count_is_rejected(self):
        generator = RandomMaskGenerator(hole_buckets=[(.333333, .333334)], bucket_probs=[1])
        with self.assertRaises(ValueError):
            generator(32, 32)

    def test_invalid_distributions_are_rejected_at_construction(self):
        for kwargs in (
            {"hole_buckets": []}, {"hole_buckets": [(0.7, 0.2)]},
            {"hole_buckets": [(-.1, .2)]}, {"hole_buckets": [(0, float("nan"))]},
            {"bucket_probs": [1]}, {"bucket_probs": [0, 0, 0, 0]},
            {"bucket_probs": [1, 1, 1, -1]}, {"bucket_probs": [1, 1, 1, float("inf")]},
            {"topology_modes": ["unknown"]}, {"topology_modes": []},
            {"topology_probs": [0, 0, 0, 0]}, {"size": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RandomMaskGenerator(**kwargs)

    def test_single_pixel_empty_and_full_buckets_have_exact_area(self):
        for ratio in (0., 1.):
            for topology in ("thin_distributed", "scattered_irregular", "large_contiguous", "mixed"):
                with self.subTest(ratio=ratio, topology=topology):
                    generator = RandomMaskGenerator(
                        hole_buckets=[(ratio, ratio)], topology_modes=[topology], max_attempts=1,
                    )
                    mask = generator(1, 1)
                    self.assertEqual(mask.item(), 1 - ratio)

    def test_fixed_ratio_non_square_masks_use_integer_area(self):
        generator = RandomMaskGenerator(hole_buckets=[(.28, .28)], max_attempts=1)
        mask = generator(4, 25)
        self.assertEqual((mask == 0).sum().item(), 28)


class ExistingMetricRegressionTests(unittest.TestCase):
    def test_constant_images_have_no_edges(self):
        for value in (0., -1., .37):
            edge = metrics._edge_map(torch.full((1, 3, 16, 16), value))
            self.assertEqual(edge.count_nonzero().item(), 0)

    def test_low_amplitude_edges_are_not_promoted_by_image_normalization(self):
        image = torch.zeros(1, 3, 16, 16)
        image[..., 8:] = .001
        self.assertEqual(metrics._edge_map(image).count_nonzero().item(), 0)


class HoleMetricTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_hole_psnr_is_not_diluted_by_known_pixels(self):
        gt = torch.zeros(1, 3, 16, 16)
        pred = torch.ones_like(gt)
        known = torch.ones(1, 1, 16, 16)
        known[..., 4:12, 4:12] = 0
        item = metrics.evaluate_per_image(pred, gt, known)[0]
        self.assertIn("hole_psnr", item)
        self.assertAlmostEqual(item["hole_psnr"], 10 * math.log10(4), places=5)
        self.assertAlmostEqual(item["psnr"], 10 * math.log10(16), places=5)
        self.assertAlmostEqual(item["hole_l1"], 1.)

    def test_ssim_averages_actual_map_at_hole_pixels(self):
        gt = torch.zeros(1, 3, 16, 16)
        pred = torch.ones_like(gt)
        known = torch.ones(1, 1, 16, 16)
        known[..., 7, 7] = 0
        item = metrics.evaluate_per_image(pred, gt, known)[0]
        self.assertIn("hole_ssim", item)
        self.assertLess(item["hole_ssim"], item["ssim"])
        self.assertGreaterEqual(item["hole_ssim"], -1)

    def test_edge_tolerance_matches_displaced_edges_but_exact_f1_does_not(self):
        gt = torch.zeros(1, 3, 16, 16)
        pred = gt.clone()
        gt[..., 8:] = 1
        pred[..., 9:] = 1
        item = metrics.evaluate_per_image(pred, gt, torch.zeros(1, 1, 16, 16))[0]
        self.assertIn("hole_edge_tolerance_f1", item)
        self.assertAlmostEqual(item["hole_edge_f1"], 0.)
        self.assertAlmostEqual(item["hole_edge_tolerance_f1"], 1.)

    def test_orientation_penalizes_sign_reversal_on_strong_gt_edges(self):
        gt = torch.zeros(1, 3, 16, 16)
        gt[..., 8:] = 1
        item = metrics.evaluate_per_image(-gt, gt, torch.zeros(1, 1, 16, 16))[0]
        self.assertIn("hole_orientation_error", item)
        self.assertAlmostEqual(item["hole_orientation_error"], math.pi, places=5)
        self.assertEqual(item["hole_orientation_count"], 48)

    def test_weak_gt_gradients_do_not_contribute_orientation(self):
        gt = torch.zeros(1, 3, 16, 16)
        gt[..., 8:] = .001
        item = metrics.evaluate_per_image(-gt, gt, torch.zeros(1, 1, 16, 16))[0]
        self.assertIn("hole_orientation_count", item)
        self.assertEqual(item["hole_orientation_count"], 0)
        self.assertEqual(item["hole_orientation_error"], 0)

    def test_empty_hole_is_finite_and_reports_no_region_support(self):
        gt = torch.zeros(1, 3, 16, 16)
        item = metrics.evaluate_per_image(torch.ones_like(gt), gt, torch.ones(1, 1, 16, 16))[0]
        self.assertIn("hole_pixels", item)
        self.assertEqual(item["hole_pixels"], 0)
        self.assertEqual(item["hole_l1"], 0)
        self.assertEqual(item["hole_ssim"], 1)
        self.assertEqual(item["hole_edge_f1"], 1)
        self.assertTrue(all(math.isfinite(v) for v in item.values()))

    def test_inner_error_is_separated_from_boundary_error(self):
        gt = torch.zeros(1, 3, 16, 16)
        pred = gt.clone()
        pred[..., 5:11, 5:11] = 1
        known = torch.ones(1, 1, 16, 16)
        known[..., 3:13, 3:13] = 0
        self.assertIn("hole_boundary_l1", metrics.evaluate_per_image(pred, gt, known)[0])
        item = metrics.evaluate_per_image(pred, gt, known, boundary_width=2)[0]
        self.assertEqual(item["hole_boundary_pixels"], 64)
        self.assertEqual(item["hole_inner_pixels"], 36)
        self.assertEqual(item["hole_boundary_l1"], 0)
        self.assertEqual(item["hole_inner_l1"], 1)

    def test_lpips_unavailable_is_explicit_and_not_a_zero_score(self):
        gt = torch.zeros(1, 3, 16, 16)
        item = metrics.evaluate_per_image(gt, gt, torch.zeros(1, 1, 16, 16))[0]
        self.assertIn("lpips_available", item)
        self.assertEqual(item["lpips_available"], 0)
        self.assertNotIn("lpips", item)
        self.assertNotIn("hole_lpips", item)

    def test_nonfinite_lpips_is_unavailable_and_omitted(self):
        # The pretrained LPIPS dependency is deliberately not downloaded in tests.
        def nonfinite_metric(pred, target):
            return pred.new_tensor(float("nan"))

        gt = torch.zeros(1, 3, 16, 16)
        item = metrics.evaluate_per_image(gt, gt, torch.zeros(1, 1, 16, 16), nonfinite_metric)[0]
        self.assertEqual(item["lpips_available"], 0)
        self.assertNotIn("lpips", item)

    def test_tolerance_does_not_match_known_edges_to_false_hole_edges(self):
        gt = torch.zeros(1, 3, 16, 16)
        gt[..., 7:] = 1
        pred = gt.clone()
        pred[..., 7] = 0
        known = torch.ones(1, 1, 16, 16)
        known[..., 7] = 0
        item = metrics.evaluate_per_image(pred, gt, known)[0]
        self.assertEqual(item["hole_edge_target_count"], 0)
        self.assertEqual(item["hole_edge_predicted_count"], 16)
        self.assertEqual(item["hole_edge_precision"], 0)
        self.assertEqual(item["hole_edge_tolerance_f1"], 0)

    def test_full_hole_is_inner_and_has_no_artificial_canvas_boundary(self):
        gt = torch.zeros(1, 3, 4, 8)
        item = metrics.evaluate_per_image(torch.ones_like(gt), gt, torch.zeros(1, 1, 4, 8))[0]
        self.assertEqual(item["hole_boundary_pixels"], 0)
        self.assertEqual(item["hole_inner_pixels"], 32)
        self.assertEqual(item["hole_inner_l1"], 1)
        self.assertTrue(all(math.isfinite(v) for v in item.values()))


class StructurePredictionMetricTests(unittest.TestCase):
    def test_structure_metrics_measure_hole_only(self):
        self.assertTrue(hasattr(metrics, "evaluate_structure_per_image"))
        gt = torch.zeros(1, 3, 16, 16)
        known = torch.ones(1, 1, 16, 16)
        known[..., 4:12, 4:12] = 0
        gradient = known.repeat(1, 6, 1, 1)
        edge = known.clone()
        item = metrics.evaluate_structure_per_image({"gradient": gradient, "edge": edge}, gt, known)[0]
        self.assertEqual(item["structure_gradient_l1"], 0)
        self.assertEqual(item["structure_edge_mae"], 0)
        self.assertEqual(item["structure_consistency_l1"], 0)
        self.assertEqual(item["structure_edge_f1"], 1)

    def test_missed_structure_gradient_direction_is_ninety_degrees(self):
        self.assertTrue(hasattr(metrics, "evaluate_structure_per_image"))
        gt = torch.zeros(1, 3, 16, 16)
        gt[..., 8:] = 1
        structure = {"gradient": torch.zeros(1, 6, 16, 16), "edge": torch.zeros(1, 1, 16, 16)}
        item = metrics.evaluate_structure_per_image(structure, gt, torch.zeros(1, 1, 16, 16))[0]
        self.assertAlmostEqual(item["structure_orientation_error"], math.pi / 2, places=5)
        self.assertEqual(item["structure_edge_f1"], 0)
        self.assertEqual(item["structure_gradient_edge_f1"], 0)


class EvaluationValidityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_nonfinite_rgb_inputs_cannot_be_dropped_from_a_perfect_summary(self):
        for name in ("pred", "gt", "M"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(input=name, value=value):
                    inputs = {"pred": torch.zeros(2, 3, 16, 16),
                              "gt": torch.zeros(2, 3, 16, 16),
                              "M": torch.ones(2, 1, 16, 16)}
                    inputs[name][1, 0, 0, 0] = value
                    with self.assertRaises((ValueError, FloatingPointError)):
                        metrics.summarize_metric_items(metrics.evaluate_per_image(**inputs))

    def test_nonfinite_structure_gradient_edge_and_logits_are_rejected(self):
        gt, known = torch.zeros(1, 3, 16, 16), torch.zeros(1, 1, 16, 16)
        for name in ("gradient", "edge", "edge_logits"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(input=name, value=value):
                    structure = {"gradient": torch.zeros(1, 6, 16, 16)}
                    structure["edge_logits" if name == "edge_logits" else "edge"] = torch.zeros_like(known)
                    structure[name][0, 0, 0, 0] = value
                    with self.assertRaises((ValueError, FloatingPointError)):
                        metrics.evaluate_structure_per_image(structure, gt, known)

    def test_inner_summary_ignores_images_without_inner_pixels(self):
        gt, pred = torch.zeros(2, 3, 32, 32), torch.ones(2, 3, 32, 32)
        known = torch.ones(2, 1, 32, 32)
        known[0, :, 15:17, :] = 0
        known[1, :, 4:28, 4:28] = 0
        items = metrics.evaluate_per_image(pred, gt, known)
        self.assertEqual(items[0]["hole_inner_pixels"], 0)
        self.assertEqual(items[1]["hole_inner_pixels"], 256)
        summary = metrics.summarize_metric_items(items)
        self.assertAlmostEqual(summary["hole_inner_psnr"], 10 * math.log10(4), places=5)
        self.assertEqual(summary["hole_inner_l1"], 1)
        self.assertEqual(summary["hole_inner_pixels"], 128)
        self.assertAlmostEqual(metrics.evaluate_batch(pred, gt, known)["hole_inner_psnr"],
                               10 * math.log10(4), places=5)

    def test_bucket_summary_ignores_empty_inner_regions_with_same_hole_area(self):
        gt, pred = torch.zeros(2, 3, 32, 32), torch.ones(2, 3, 32, 32)
        known = torch.ones(2, 1, 32, 32)
        known[0, :, 1:5, :] = 0
        known[0, :, 17:21, :] = 0
        known[1, :, 8:24, 8:24] = 0
        items = metrics.evaluate_per_image(pred, gt, known)
        bucket = next(row for row in metrics.summarize_bucket_metrics(items) if row["bucket"] == "20-30%")
        self.assertEqual(bucket["num_samples"], 2)
        self.assertEqual(bucket["hole_inner_pixels"], 32)
        self.assertEqual(bucket["hole_inner_l1"], 1)
        self.assertAlmostEqual(bucket["hole_inner_psnr"], 10 * math.log10(4), places=5)

    def test_hole_and_boundary_scores_ignore_empty_regions_but_keep_counts(self):
        gt, pred = torch.zeros(2, 3, 32, 32), torch.ones(2, 3, 32, 32)
        known = torch.ones(2, 1, 32, 32)
        known[1, :, 4:28, 4:28] = 0
        items = metrics.evaluate_per_image(pred, gt, known)
        summary = metrics.summarize_metric_items(items)
        self.assertEqual(summary["hole_l1"], 1)
        self.assertEqual(summary["hole_boundary_l1"], 1)
        self.assertEqual(summary["hole_pixels"], 288)
        self.assertEqual(summary["hole_edge_f1"], 0)

    def test_orientation_summary_uses_only_strong_edge_support(self):
        gt = torch.zeros(2, 3, 16, 16)
        gt[1, ..., 8:] = 1
        known = torch.zeros(2, 1, 16, 16)
        items = metrics.evaluate_per_image(-gt, gt, known)
        summary = metrics.summarize_metric_items(items)
        self.assertAlmostEqual(summary["hole_orientation_error"], math.pi, places=5)
        self.assertEqual(summary["hole_orientation_count"], 24)
        structure = {"gradient": torch.zeros(2, 6, 16, 16), "edge": torch.zeros_like(known)}
        structure_summary = metrics.summarize_metric_items(metrics.evaluate_structure_per_image(structure, gt, known))
        self.assertAlmostEqual(structure_summary["structure_orientation_error"], math.pi / 2, places=5)
        self.assertEqual(structure_summary["structure_orientation_count"], 24)

    def test_empty_edge_sets_with_nonempty_holes_remain_valid_perfect_scores(self):
        gt = torch.zeros(1, 3, 16, 16)
        known = torch.zeros(1, 1, 16, 16)
        item = metrics.evaluate_per_image(gt, gt, known)[0]
        summary = metrics.summarize_metric_items([item])
        self.assertEqual(summary["hole_edge_f1"], 1)
        self.assertNotIn("hole_orientation_error", summary)
        self.assertNotIn("hole_boundary_l1", summary)
        self.assertEqual(summary["hole_orientation_count"], 0)


if __name__ == "__main__":
    unittest.main()
