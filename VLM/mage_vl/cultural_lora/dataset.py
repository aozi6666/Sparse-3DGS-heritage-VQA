"""Dataset loaders for Cultural LoRA (CiQi images + MSVD frame packs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths import (
    CIQI_SUBSET_JSONL,
    DEFAULT_CIQI_DIR,
    DEFAULT_MSVD_DIR,
    MSVD_SUBSET_DIR,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class CiQiSubsetDataset(Dataset):
    def __init__(self, ciqi_dir: Path | None = None, jsonl_name: str = CIQI_SUBSET_JSONL):
        self.root = Path(ciqi_dir or DEFAULT_CIQI_DIR)
        path = self.root / jsonl_name
        if not path.is_file():
            # fallback to train.jsonl after hard-replace
            path = self.root / "train.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"CiQi jsonl not found under {self.root}")
        self.rows = _load_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.rows[idx]
        images = []
        for rel in rec.get("images") or []:
            p = self.root / rel
            images.append(Image.open(p).convert("RGB"))
        messages = rec["messages"]
        # Extract user text + assistant text from content blocks
        user_text = ""
        answer = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            texts = []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text") or "")
            elif isinstance(content, str):
                texts.append(content)
            joined = "\n".join(texts).strip()
            if role == "user":
                user_text = joined
            elif role == "assistant":
                answer = joined
        return {
            "modality": "image",
            "images": images,
            "question": user_text,
            "answer": answer,
            "id": rec.get("id"),
        }


class MSVDSubsetDataset(Dataset):
    def __init__(
        self,
        msvd_dir: Path | None = None,
        max_frames: int = 8,
    ):
        self.root = Path(msvd_dir or DEFAULT_MSVD_DIR) / MSVD_SUBSET_DIR
        qa_path = self.root / "qa.jsonl"
        if not qa_path.is_file():
            raise FileNotFoundError(f"MSVD subset missing: {qa_path}")
        self.rows = _load_jsonl(qa_path)
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.rows[idx]
        frames = np.load(self.root / rec["frames_file"])
        if frames.ndim != 4:
            raise ValueError(f"bad frames shape {frames.shape} for {rec['frames_file']}")
        t = frames.shape[0]
        if t > self.max_frames:
            sel = np.linspace(0, t - 1, self.max_frames, dtype=int)
            frames = frames[sel]
        pil_frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        return {
            "modality": "video",
            "frames": pil_frames,
            "question": rec["question"],
            "answer": rec["answer"],
            "id": rec.get("video_key"),
        }


class MixedCulturalDataset(Dataset):
    """Concatenate CiQi + MSVD samples for joint Cultural LoRA."""

    def __init__(
        self,
        ciqi_dir: Path | None = None,
        msvd_dir: Path | None = None,
        max_frames: int = 8,
        include_ciqi: bool = True,
        include_msvd: bool = True,
    ):
        self.parts: list[Dataset] = []
        self.offsets: list[int] = []
        total = 0
        if include_ciqi:
            d = CiQiSubsetDataset(ciqi_dir)
            self.parts.append(d)
            self.offsets.append(total)
            total += len(d)
        if include_msvd:
            d = MSVDSubsetDataset(msvd_dir, max_frames=max_frames)
            self.parts.append(d)
            self.offsets.append(total)
            total += len(d)
        if not self.parts:
            raise ValueError("no datasets enabled")
        self._len = total

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)
        for i in range(len(self.offsets) - 1, -1, -1):
            if idx >= self.offsets[i]:
                return self.parts[i][idx - self.offsets[i]]
        raise IndexError(idx)


def _clip_answer(text: str, max_answer_chars: int) -> str:
    if max_answer_chars <= 0 or len(text) <= max_answer_chars:
        return text
    # Prefer cutting at a paragraph / sentence boundary near the cap
    cut = text[:max_answer_chars]
    for sep in ("\n\n", "。", "！", "？", ".", "!", "?"):
        pos = cut.rfind(sep)
        if pos >= max_answer_chars // 2:
            return cut[: pos + len(sep)].rstrip()
    return cut.rstrip() + "…"


def build_chat_messages(
    sample: dict[str, Any],
    max_answer_chars: int = 1024,
    ciqi_max_images: int = 2,
    num_images: int | None = None,
) -> list[dict]:
    """Build chat; `num_images` overrides placeholder count (must match PIL list length)."""
    if num_images is None:
        if sample["modality"] == "image":
            num_images = min(len(sample["images"]), max(1, ciqi_max_images))
        else:
            num_images = len(sample["frames"])
    content: list[dict] = [{"type": "image"} for _ in range(max(1, int(num_images)))]
    content.append({"type": "text", "text": sample["question"]})
    answer = _clip_answer(sample["answer"] or "", max_answer_chars)
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def sample_images(
    sample: dict[str, Any],
    ciqi_max_images: int = 2,
) -> list[Image.Image]:
    if sample["modality"] == "image":
        imgs = sample["images"]
        return imgs[: max(1, ciqi_max_images)]
    return list(sample["frames"])


def subsample_pil_list(images: list[Image.Image], n: int) -> list[Image.Image]:
    """Uniformly keep `n` frames/images (n>=1). Preserves temporal coverage for video."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(images) <= n:
        return list(images)
    import numpy as np

    idx = np.linspace(0, len(images) - 1, n, dtype=int)
    # unique while preserving order
    seen = set()
    out: list[Image.Image] = []
    for i in idx.tolist():
        if i not in seen:
            seen.add(i)
            out.append(images[i])
    return out if out else [images[0]]
