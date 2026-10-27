"""Mage-VL image generate helper for OCRBench image QueryIP (new file; no video edits)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .mage_forward import MageQueryIPRunner, ensure_import_paths


@dataclass
class ImageForwardStats:
    mode: str
    question: str
    answer: str
    n_vis_tokens: int
    wall_s: float
    extra: dict[str, Any]


class ImageMageRunner:
    """Thin image generate API reusing MageQueryIPRunner weights/processor."""

    def __init__(self, model_path: str | Path, local_files_only: bool = True):
        self.inner = MageQueryIPRunner(model_path, local_files_only=local_files_only)

    def generate_image(
        self,
        image: Image.Image,
        question: str,
        *,
        mode: str = "full_image",
        max_new_tokens: int = 128,
        max_pixels: int = 150000,
        extra: dict[str, Any] | None = None,
    ) -> ImageForwardStats:
        ensure_import_paths(self.inner.model_path)
        t0 = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.inner.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pil = image.convert("RGB")
        # Prefer max_pixels when processor supports it; fall back otherwise.
        try:
            inputs = self.inner.processor(
                text=[text],
                images=[pil],
                return_tensors="pt",
                max_pixels=int(max_pixels),
            )
        except TypeError:
            inputs = self.inner.processor(
                text=[text], images=[pil], return_tensors="pt"
            )
        moved = {
            k: (v.to(self.inner.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        if "pixel_values" in moved:
            moved["pixel_values"] = moved["pixel_values"].to(self.inner.model.dtype)

        n_vis = 0
        if "image_grid_thw" in moved:
            grid = moved["image_grid_thw"]
            n_vis = int(grid.prod().item()) if hasattr(grid, "prod") else 0
            sms = int(getattr(self.inner.processor, "spatial_merge_size", 2) or 2)
            n_vis = n_vis // max(1, sms * sms)

        with torch.inference_mode():
            out = self.inner.model.generate(
                **moved, max_new_tokens=int(max_new_tokens), do_sample=False
            )
        answer = self.inner.processor.tokenizer.decode(
            out[0, moved["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        wall = time.perf_counter() - t0
        return ImageForwardStats(
            mode=mode,
            question=question,
            answer=answer,
            n_vis_tokens=n_vis,
            wall_s=wall,
            extra=dict(extra or {}),
        )
