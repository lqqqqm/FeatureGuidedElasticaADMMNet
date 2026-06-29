from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


class RandomMaskGenerator:
    """Generate binary masks with M=1 known, M=0 hole."""

    def __init__(
        self,
        size: int = 256,
        hole_buckets: Optional[Sequence[Sequence[float]]] = None,
        bucket_probs: Optional[Sequence[float]] = None,
        topology_modes: Optional[Sequence[str]] = None,
        topology_probs: Optional[Sequence[float]] = None,
    ):
        self.size = size
        self.hole_buckets = list(hole_buckets or [(0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.6)])
        self.bucket_probs = list(bucket_probs or [0.25, 0.25, 0.25, 0.25])
        self.topology_modes = list(topology_modes or ["thin_distributed", "scattered_irregular", "large_contiguous", "mixed"])
        self.topology_probs = list(topology_probs or [0.30, 0.30, 0.25, 0.15])

    @staticmethod
    def _hole_ratio(mask_np: np.ndarray) -> float:
        return 1.0 - float(mask_np.mean())

    def _draw_rectangles(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        for _ in range(random.randint(1, 5)):
            x1 = random.randint(0, max(w - 16, 1))
            y1 = random.randint(0, max(h - 16, 1))
            x2 = random.randint(x1 + 8, min(w, x1 + max(w // 2, 16)))
            y2 = random.randint(y1 + 8, min(h, y1 + max(h // 2, 16)))
            draw.rectangle([x1, y1, x2, y2], fill=0)

    def _draw_brushes(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        num_strokes = random.randint(4, 12)
        for _ in range(num_strokes):
            pts = []
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            for _ in range(random.randint(3, 8)):
                pts.append((x, y))
                x = int(np.clip(x + random.randint(-40, 40), 0, w - 1))
                y = int(np.clip(y + random.randint(-40, 40), 0, h - 1))
            width = random.randint(8, 32)
            draw.line(pts, fill=0, width=width)
            r = max(width // 2, 1)
            for px, py in pts:
                draw.ellipse([px - r, py - r, px + r, py + r], fill=0)

    def _draw_thin_distributed(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        for _ in range(random.randint(10, 24)):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            pts = []
            for _ in range(random.randint(2, 5)):
                pts.append((x, y))
                x = int(np.clip(x + random.randint(-50, 50), 0, w - 1))
                y = int(np.clip(y + random.randint(-50, 50), 0, h - 1))
            draw.line(pts, fill=0, width=random.randint(2, 8))

    def _draw_scattered_irregular(self, draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
        for _ in range(random.randint(6, 18)):
            cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
            rx, ry = random.randint(5, max(w // 10, 6)), random.randint(5, max(h // 10, 6))
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
        x1 = random.randint(0, max(w // 2, 1))
        y1 = random.randint(0, max(h // 2, 1))
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
        low, high = random.choices(self.hole_buckets, weights=self.bucket_probs, k=1)[0]
        topology = random.choices(self.topology_modes, weights=self.topology_probs, k=1)[0]
        mask = None
        for _ in range(50):
            candidate = self._sample_canvas(h, w, topology)
            ratio = self._hole_ratio(candidate)
            if low <= ratio <= high:
                mask = candidate
                break
            mask = candidate
        return torch.from_numpy(mask).unsqueeze(0)


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
