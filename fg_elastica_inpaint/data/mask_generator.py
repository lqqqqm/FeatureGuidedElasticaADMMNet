from __future__ import annotations

import math
import random
from numbers import Integral
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


class RandomMaskGenerator:
    """Generate binary masks with M=1 known, M=0 hole.

    Choose a bucket and topology once. After bounded rejection sampling, grow
    or erode the closest candidate along its boundary to an attainable count
    in that bucket. This keeps the drawn layout as far as possible; extreme
    ratios can merge components or thicken thin strokes. An unrepresentable
    bucket raises ValueError instead of silently changing the distribution.
    """

    def __init__(
        self,
        size: int = 256,
        hole_buckets: Optional[Sequence[Sequence[float]]] = None,
        bucket_probs: Optional[Sequence[float]] = None,
        topology_modes: Optional[Sequence[str]] = None,
        topology_probs: Optional[Sequence[float]] = None,
        max_attempts: int = 50,
    ):
        self.size = self._positive_integer(size, "size")
        self.max_attempts = self._positive_integer(max_attempts, "max_attempts")
        self.hole_buckets = list(hole_buckets if hole_buckets is not None else
                                 [(0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.6)])
        if not self.hole_buckets:
            raise ValueError("hole_buckets must not be empty.")
        for bucket in self.hole_buckets:
            if len(bucket) != 2 or not all(math.isfinite(float(v)) for v in bucket):
                raise ValueError("Each hole bucket needs two finite bounds.")
            low, high = bucket
            if not 0 <= low <= high <= 1:
                raise ValueError("Hole bucket bounds must satisfy 0 <= low <= high <= 1.")
        self.bucket_probs = self._weights(bucket_probs, len(self.hole_buckets), "bucket_probs")
        modes = ["thin_distributed", "scattered_irregular", "large_contiguous", "mixed"]
        self.topology_modes = list(modes if topology_modes is None else topology_modes)
        if not self.topology_modes or any(mode not in modes for mode in self.topology_modes):
            raise ValueError(f"topology_modes must be a nonempty subset of {modes}.")
        if topology_probs is None and topology_modes is None:
            topology_probs = [0.30, 0.30, 0.25, 0.15]
        self.topology_probs = self._weights(topology_probs, len(self.topology_modes), "topology_probs")

    @staticmethod
    def _positive_integer(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    @staticmethod
    def _weights(values: Optional[Sequence[float]], count: int, name: str) -> list[float]:
        weights = [1.0] * count if values is None else [float(v) for v in values]
        if (len(weights) != count or any(not math.isfinite(v) or v < 0 for v in weights)
                or not math.isfinite(sum(weights)) or sum(weights) <= 0):
            raise ValueError(f"{name} must have {count} finite nonnegative weights with positive sum.")
        return weights

    @staticmethod
    def _hole_ratio(mask_np: np.ndarray) -> float:
        return float(np.count_nonzero(mask_np == 0)) / mask_np.size

    def _draw_rectangles(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        for _ in range(random.randint(1, 5)):
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)
            x2 = min(w - 1, x1 + random.randint(max(w // 32, 1), max(w // 2, 1)))
            y2 = min(h - 1, y1 + random.randint(max(h // 32, 1), max(h // 2, 1)))
            draw.rectangle([x1, y1, x2, y2], fill=0)

    def _draw_brushes(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        num_strokes = random.randint(4, 12)
        step_x, step_y = max(round(w * 40 / 256), 1), max(round(h * 40 / 256), 1)
        for _ in range(num_strokes):
            pts = []
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            for _ in range(random.randint(3, 8)):
                pts.append((x, y))
                x = int(np.clip(x + random.randint(-step_x, step_x), 0, w - 1))
                y = int(np.clip(y + random.randint(-step_y, step_y), 0, h - 1))
            width = random.randint(max(min(w, h) // 32, 1), max(min(w, h) // 8, 1))
            draw.line(pts, fill=0, width=width)
            r = max(width // 2, 1)
            for px, py in pts:
                draw.ellipse([px - r, py - r, px + r, py + r], fill=0)

    def _draw_thin_distributed(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        step_x, step_y = max(round(w * 50 / 256), 1), max(round(h * 50 / 256), 1)
        for _ in range(random.randint(10, 24)):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            pts = []
            for _ in range(random.randint(2, 5)):
                pts.append((x, y))
                x = int(np.clip(x + random.randint(-step_x, step_x), 0, w - 1))
                y = int(np.clip(y + random.randint(-step_y, step_y), 0, h - 1))
            draw.line(pts, fill=0, width=random.randint(max(min(w, h) // 128, 1), max(min(w, h) // 32, 1)))

    def _draw_scattered_irregular(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        for _ in range(random.randint(6, 18)):
            cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
            rx = random.randint(max(round(w * 5 / 256), 1), max(w // 10, 1))
            ry = random.randint(max(round(h * 5 / 256), 1), max(h // 10, 1))
            points = []
            for t in np.linspace(0, 2 * np.pi, random.randint(6, 12), endpoint=False):
                jitter = random.uniform(0.5, 1.2)
                px = int(np.clip(cx + np.cos(t) * rx * jitter, 0, w - 1))
                py = int(np.clip(cy + np.sin(t) * ry * jitter, 0, h - 1))
                points.append((px, py))
            draw.polygon(points, fill=0)

    def _draw_large_contiguous(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        if random.random() < 0.5:
            self._draw_free_form(draw, w, h)
            return
        x1 = random.randint(0, min(w - 1, w // 2))
        y1 = random.randint(0, min(h - 1, h // 2))
        x2 = random.randint(max(x1 + w // 4, x1 + 1), w)
        y2 = random.randint(max(y1 + h // 4, y1 + 1), h)
        draw.rectangle([x1, y1, x2, y2], fill=0)

    def _draw_free_form(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        num_vertices = random.randint(6, 14)
        points = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(num_vertices)]
        draw.polygon(points, fill=0)

    def _sample_canvas(self, h: int, w: int, topology: str) -> np.ndarray:
        canvas = Image.new("L", (w, h), color=255)
        draw = ImageDraw.Draw(canvas)
        if topology == "thin_distributed":
            self._draw_thin_distributed(draw, w, h)
            return np.asarray(canvas, dtype=np.float32) / 255.0
        if topology == "scattered_irregular":
            self._draw_scattered_irregular(draw, w, h)
            return np.asarray(canvas, dtype=np.float32) / 255.0
        if topology == "large_contiguous":
            self._draw_large_contiguous(draw, w, h)
            return np.asarray(canvas, dtype=np.float32) / 255.0

        mode = random.choice(["rect", "brush", "mix", "free"])
        if mode in ["rect", "mix"]:
            self._draw_rectangles(draw, w, h)
        if mode in ["brush", "mix"]:
            self._draw_brushes(draw, w, h)
        if mode == "free":
            self._draw_free_form(draw, w, h)
        return np.asarray(canvas, dtype=np.float32) / 255.0

    def __call__(self, h: int, w: int) -> torch.Tensor:
        h, w = self._positive_integer(h, "h"), self._positive_integer(w, "w")
        low, high = random.choices(self.hole_buckets, weights=self.bucket_probs, k=1)[0]
        topology = random.choices(self.topology_modes, weights=self.topology_probs, k=1)[0]
        n = h * w
        minimum, maximum = math.ceil(low * n), math.floor(high * n)
        # Compare ratios too, avoiding off-by-one caused by floating products.
        if minimum > 0 and (minimum - 1) / n >= low:
            minimum -= 1
        if maximum < n and (maximum + 1) / n <= high:
            maximum += 1
        if minimum > maximum:
            raise ValueError(f"Bucket [{low}, {high}] contains no pixel count for {h}x{w}.")
        mask, best_distance = None, n + 1
        for _ in range(self.max_attempts):
            candidate = self._sample_canvas(h, w, topology)
            count = int(np.count_nonzero(candidate == 0))
            if minimum <= count <= maximum:
                return torch.from_numpy(candidate.copy()).unsqueeze(0)
            distance = max(minimum - count, count - maximum)
            if distance < best_distance:
                mask, best_distance = candidate, distance
        target = random.randint(minimum, maximum)
        mask = self._adjust_hole_count(mask, target)
        if not low <= self._hole_ratio(mask) <= high:
            raise RuntimeError("Mask adjustment failed to honor the sampled hole bucket.")
        return torch.from_numpy(mask).unsqueeze(0)

    @staticmethod
    def _adjust_hole_count(mask: np.ndarray, target: int) -> np.ndarray:
        """Modify boundary layers, with a partial last layer for exact area.

        Uses the Python RNG, so the existing dataloader worker seeding remains
        sufficient. Outside the canvas is known during erosion. At least one
        pixel changes each round, giving a finite bound independent of luck.
        """
        hole = mask == 0
        count = int(hole.sum())
        for _ in range(hole.size + 1):
            if count == target:
                return (~hole).astype(np.float32)
            padded = np.pad(hole, 1, constant_values=False)
            neighbors = [padded[y:y + hole.shape[0], x:x + hole.shape[1]]
                         for y in range(3) for x in range(3)]
            grow = count < target
            if grow:
                boundary = np.logical_or.reduce(neighbors) & ~hole
            else:
                boundary = hole & ~np.logical_and.reduce(neighbors)
            choices = np.flatnonzero(boundary)
            if not choices.size:
                # An empty seed can only occur when a drawing erases no pixel.
                choices = np.flatnonzero(~hole if grow else hole)
            amount = min(abs(target - count), len(choices))
            chosen = choices if amount == len(choices) else random.sample(choices.tolist(), amount)
            hole.flat[chosen] = grow
            count += amount if grow else -amount
        raise RuntimeError("Could not adjust mask to requested hole count.")


class FixedMaskLoader:
    def __init__(self, mask_paths: Sequence[str | Path], image_size: int = 256):
        self.mask_paths = [str(p) for p in mask_paths]
        self.image_size = image_size
        if not self.mask_paths:
            raise ValueError("FixedMaskLoader needs at least one mask path.")

    def __len__(self) -> int:
        return len(self.mask_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.mask_paths[index % len(self.mask_paths)]
        mask = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        return torch.from_numpy(mask_np).unsqueeze(0)
