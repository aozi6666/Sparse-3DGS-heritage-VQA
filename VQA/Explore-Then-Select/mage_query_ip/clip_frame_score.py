"""CLIP 帧–问题余弦相关（方案 A）。

【主路径不用】E4 失败消融：queryip_clip=58.5% < random。保留仅供复现/对照。
禁止再作为选区主卖点；勿用于 16×16 单 patch 打分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

# Default local weight (repo-relative).
_DEFAULT_CLIP = (
    Path(__file__).resolve().parents[2] / "models" / "CLIP-ViT-Large-patch14"
)


def _clip_feat(out: torch.Tensor | object) -> torch.Tensor:
    """transformers≥5: get_*_features → BaseModelOutputWithPooling; older → Tensor."""
    if hasattr(out, "pooler_output"):
        out = out.pooler_output
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"expected CLIP feature tensor, got {type(out)}")
    return out


class ClipFrameScorer:
    """Lazy-load CLIP; score RGB frames against a text query on CPU or CUDA."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path else _DEFAULT_CLIP
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self.local_files_only = bool(local_files_only)
        self._model = None
        self._processor = None
        self._text_cache: dict[str, torch.Tensor] = {}

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"CLIP weights not found: {self.model_path}")
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(
            str(self.model_path), local_files_only=self.local_files_only
        )
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = CLIPModel.from_pretrained(
            str(self.model_path),
            local_files_only=self.local_files_only,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self._model.to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        self._ensure()
        key = text.strip()
        if key in self._text_cache:
            return self._text_cache[key]
        assert self._processor is not None and self._model is not None
        inputs = self._processor(
            text=[key], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = _clip_feat(self._model.get_text_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        self._text_cache[key] = feat
        return feat

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        self._ensure()
        assert self._processor is not None and self._model is not None
        if not images:
            raise ValueError("no images")
        rgb = [im.convert("RGB") for im in images]
        inputs = self._processor(images=rgb, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feat = _clip_feat(self._model.get_image_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def score_frames(
        self,
        images: Sequence[Image.Image],
        question: str,
        *,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return q_t cosine similarities, shape (n_frames,)."""
        text_f = self.encode_text(question)
        qs: list[float] = []
        for i in range(0, len(images), int(batch_size)):
            batch = list(images[i : i + int(batch_size)])
            img_f = self.encode_images(batch)
            sim = (img_f * text_f).sum(dim=-1)
            qs.extend(sim.detach().float().cpu().tolist())
        return np.asarray(qs, dtype=np.float32)


def sample_video_frames(
    video_path: str | Path,
    num_frames: int = 8,
) -> list[Image.Image]:
    """Uniformly sample RGB frames from a video (decord → cv2 → imageio)."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    n = max(1, int(num_frames))

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        total = len(vr)
        if total <= 0:
            raise RuntimeError("empty video")
        idxs = np.linspace(0, total - 1, num=min(n, total), dtype=int)
        batch = vr.get_batch(idxs.tolist()).asnumpy()
        return [Image.fromarray(batch[i], mode="RGB") for i in range(batch.shape[0])]
    except Exception:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            raise RuntimeError("cv2 empty")
        idxs = set(np.linspace(0, total - 1, num=min(n, total), dtype=int).tolist())
        out: list[Image.Image] = []
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fi in idxs:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.append(Image.fromarray(rgb, mode="RGB"))
            fi += 1
        cap.release()
        if out:
            return out
    except Exception:
        pass

    raise RuntimeError(
        f"could not sample frames from {path}; install decord or opencv-python"
    )
