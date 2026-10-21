"""Mage-VL Query-IP adapter for VLMEvalKit (question-conditioned codec reweight)."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from .base import BaseModel

DEFAULT_MODEL_PATH = "/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/mage_vl/models"
_REPO_ROOT = Path("/root/autodl-tmp/Sparse-3DGS-heritage-VQA")
_ETS_ROOT = _REPO_ROOT / "VQA" / "Explore-Then-Select"
DEFAULT_TOKEN_LOG = str(
    _REPO_ROOT / "VQA" / "eval_results" / "Mage-VL-QueryIP_videomme_tokens.jsonl"
)


def _ensure_mage_query_ip_importable(model_path: str) -> None:
    for p in (_ETS_ROOT, Path(model_path)):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


class MageVLQueryIPChat(BaseModel):
    """VLMEvalKit wrapper around MageQueryIPRunner.run_reweight (fixed keep_ratio budget)."""

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        nframe: int = 32,
        max_pixels: int = 150000,
        max_new_tokens: int = 256,
        keep_ratio: float = 0.75,
        codec_engine: str = "traditional",
        token_log_path: str | None = DEFAULT_TOKEN_LOG,
        local_files_only: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert model_path is not None
        self.model_path = model_path
        self.nframe = int(nframe)
        self.max_pixels = int(max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.keep_ratio = float(keep_ratio)
        self.codec_engine = "hevc" if codec_engine == "traditional" else str(codec_engine)
        self.token_log_path = token_log_path

        if kwargs:
            warnings.warn(f"Unused MageVLQueryIPChat kwargs ignored: {sorted(kwargs.keys())}")

        _ensure_mage_query_ip_importable(self.model_path)
        from mage_query_ip.mage_forward import MageQueryIPRunner

        self.runner = MageQueryIPRunner(
            self.model_path, local_files_only=local_files_only
        )

    @staticmethod
    def _parse_message(message):
        images = []
        videos = []
        texts = []
        for item in message:
            typ = item["type"]
            val = item["value"]
            if typ == "image":
                images.append(val)
            elif typ == "video":
                videos.append(val)
            elif typ == "text":
                texts.append(val)
            else:
                raise ValueError(f"Unsupported message type for Mage-VL-QueryIP: {typ}")
        question = "\n".join(texts).strip()
        if not question:
            question = "Describe this media."
        return images, videos, question

    def _append_token_log(self, record: dict) -> None:
        if not self.token_log_path:
            return
        path = Path(self.token_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _json_default(o):
            if hasattr(o, "item"):
                return o.item()
            return str(o)

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def generate_inner(self, message, dataset=None):
        images, videos, question = self._parse_message(message)
        if images:
            raise ValueError("Mage-VL-QueryIP adapter supports video only (no images).")
        if len(videos) != 1:
            raise ValueError("Mage-VL-QueryIP adapter requires exactly one video.")

        video_path = videos[0]
        budget = self.runner.measure_default_patches(
            video=video_path,
            num_frames=self.nframe,
            max_pixels=self.max_pixels,
            engine=self.codec_engine,
        )
        budget_k = max(1, int(round(budget.n_patches * self.keep_ratio)))
        stats = self.runner.run_reweight(
            video=video_path,
            question=question,
            num_frames=self.nframe,
            max_pixels=self.max_pixels,
            max_new_tokens=self.max_new_tokens,
            engine=self.codec_engine,
            budget_k=budget_k,
            keep_ratio=self.keep_ratio,
        )
        self._append_token_log(
            {
                "dataset": dataset,
                "video": video_path,
                "question": question,
                "qtype": stats.qtype,
                "n_patches_default": budget.n_patches,
                "n_vis_tokens_default": budget.n_vis_tokens,
                "budget_k": budget_k,
                "keep_ratio": self.keep_ratio,
                "n_patches": stats.n_patches,
                "n_vis_tokens": stats.n_vis_tokens,
                "wall_s": stats.wall_s,
                "measure_wall_s": budget.wall_s,
                "alpha": (stats.extra or {}).get("alpha"),
                "beta": (stats.extra or {}).get("beta"),
                "mean_motion_kept": (stats.extra or {}).get("mean_motion_kept"),
            }
        )
        return stats.answer
