"""Mage-VL forward wrapper with optional query-IP reweight (does not edit mage_vl/)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .codec_reweight import (
    ReweightResult,
    SelectMode,
    reweight_codec_canvases,
    select_patches_samek,
)
from .question_type import QuestionType, classify_question

# .../VQA/Explore-Then-Select/mage_query_ip/mage_forward.py → repo root = parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ETS_ROOT = Path(__file__).resolve().parents[1]
_MAGE_MODELS = _REPO_ROOT / "VQA" / "mage_vl" / "models"


def ensure_import_paths(model_path: str | Path | None = None) -> None:
    """Inject mage_vl/models + Explore-Then-Select onto sys.path (idempotent)."""
    candidates = [_ETS_ROOT, _MAGE_MODELS]
    if model_path is not None:
        candidates.append(Path(model_path))
    for p in candidates:
        s = str(p.resolve())
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


@dataclass
class ForwardStats:
    mode: str
    question: str
    qtype: Optional[str]
    n_vis_tokens: int
    n_patches: int
    wall_s: float
    answer: str
    reweight: Optional[ReweightResult] = None
    extra: dict[str, Any] | None = None


@dataclass
class PatchBudget:
    """Codec-only default patch count (no LLM)."""

    n_patches: int
    n_canvas: int
    n_vis_tokens: int
    wall_s: float


def _import_codec():
    ensure_import_paths()
    try:
        from codec_video_processing_mage_vl import (
            CodecConfig,
            process_codec_video,
            drop_padding_canvases,
            codec_positions_for_processor,
            rewrite_text_with_codec_positions,
            codec_image_processor_outputs,
        )
    except ImportError as e:
        raise ImportError(
            "Cannot import Mage codec helpers. Set PYTHONPATH to include "
            "VQA/mage_vl/models (see COMMANDS.md)."
        ) from e
    return (
        CodecConfig,
        process_codec_video,
        drop_padding_canvases,
        codec_positions_for_processor,
        rewrite_text_with_codec_positions,
        codec_image_processor_outputs,
    )


class MageQueryIPRunner:
    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        torch_dtype: str = "auto",
        local_files_only: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor

        ensure_import_paths(model_path)
        self.model_path = str(model_path)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
            local_files_only=local_files_only,
        ).eval()
        self.device = device or next(self.model.parameters()).device

    def _codec_cfg(
        self,
        num_frames: int = 32,
        max_pixels: int = 150000,
        engine: str = "hevc",
        patch: int = 16,
    ):
        CodecConfig, *_ = _import_codec()
        return CodecConfig(
            engine=engine,
            target_canvas=int(num_frames),
            patch=int(patch),
            max_pixels=int(max_pixels),
        )

    def _build_inputs_from_codec_payload(
        self,
        question: str,
        images,
        src_positions,
        fps: float,
        max_pixels: int,
    ) -> tuple[dict, int]:
        (
            _,
            _,
            _,
            codec_positions_for_processor,
            rewrite_text_with_codec_positions,
            codec_image_processor_outputs,
        ) = _import_codec()

        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": question},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_data = codec_image_processor_outputs(
            self.processor.image_processor, images, max_pixels=int(max_pixels)
        )
        image_grid_thw = image_data["image_grid_thw"]
        patch_positions = codec_positions_for_processor(
            src_positions, image_grid_thw, device=image_grid_thw.device
        )
        text = rewrite_text_with_codec_positions(
            text, patch_positions, fps=float(fps), decimals=1
        )
        encoding = self.processor.tokenizer(
            [text], padding=True, return_tensors="pt"
        )
        inputs = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding.get(
                "attention_mask", torch.ones_like(encoding["input_ids"])
            ),
            "pixel_values": image_data["pixel_values"],
            "image_grid_thw": image_grid_thw,
            "patch_positions": patch_positions,
        }
        n_patches = int(patch_positions.shape[0])
        # After spatial merge, visual tokens ≈ n_patches / merge^2
        sms = int(getattr(self.processor, "spatial_merge_size", 2) or 2)
        n_vis = n_patches // (sms * sms)
        return inputs, n_vis

    def _generate(self, inputs: dict, max_new_tokens: int) -> str:
        moved = {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }
        if "pixel_values" in moved:
            moved["pixel_values"] = moved["pixel_values"].to(self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(
                **moved, max_new_tokens=int(max_new_tokens), do_sample=False
            )
        return self.processor.tokenizer.decode(
            out[0, moved["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def measure_default_patches(
        self,
        video: str,
        num_frames: int = 32,
        max_pixels: int = 150000,
        engine: str = "hevc",
    ) -> PatchBudget:
        """Codec + drop_padding only; used to set budget_k without LLM generate."""
        (
            _,
            process_codec_video,
            drop_padding_canvases,
            *_,
        ) = _import_codec()
        cfg = self._codec_cfg(num_frames=num_frames, max_pixels=max_pixels, engine=engine)
        t0 = time.perf_counter()
        payload = process_codec_video(video, cfg)
        imgs, src_pos, _ = drop_padding_canvases(payload["images"], payload["src_positions"])
        n_patches = int(src_pos.shape[0])
        sms = int(getattr(self.processor, "spatial_merge_size", 2) or 2)
        n_vis = n_patches // (sms * sms)
        return PatchBudget(
            n_patches=n_patches,
            n_canvas=len(imgs),
            n_vis_tokens=n_vis,
            wall_s=time.perf_counter() - t0,
        )

    def run_default(
        self,
        video: str,
        question: str,
        num_frames: int = 32,
        max_pixels: int = 150000,
        max_new_tokens: int = 64,
        engine: str = "hevc",
    ) -> ForwardStats:
        (
            _,
            process_codec_video,
            drop_padding_canvases,
            *_,
        ) = _import_codec()
        cfg = self._codec_cfg(num_frames=num_frames, max_pixels=max_pixels, engine=engine)
        t0 = time.perf_counter()
        payload = process_codec_video(video, cfg)
        imgs, src_pos, _ = drop_padding_canvases(payload["images"], payload["src_positions"])
        inputs, n_vis = self._build_inputs_from_codec_payload(
            question, imgs, src_pos, payload["fps"], max_pixels
        )
        answer = self._generate(inputs, max_new_tokens)
        wall = time.perf_counter() - t0
        return ForwardStats(
            mode="default",
            question=question,
            qtype=None,
            n_vis_tokens=n_vis,
            n_patches=int(src_pos.shape[0]),
            wall_s=wall,
            answer=answer,
            extra={"n_canvas": len(imgs)},
        )

    def encode_codec(
        self,
        video: str,
        num_frames: int = 32,
        max_pixels: int = 150000,
        engine: str = "hevc",
    ) -> dict[str, Any]:
        """Run codec once; reuse payload for same-K multi-mode ablations."""
        (
            _,
            process_codec_video,
            drop_padding_canvases,
            *_,
        ) = _import_codec()
        cfg = self._codec_cfg(num_frames=num_frames, max_pixels=max_pixels, engine=engine)
        payload = process_codec_video(video, cfg)
        imgs, src_pos, _ = drop_padding_canvases(payload["images"], payload["src_positions"])
        return {
            "video": str(video),
            "images": imgs,
            "src_positions": src_pos,
            "fps": float(payload["fps"]),
            "meta": payload.get("meta") or {},
            "n_patches": int(src_pos.shape[0]),
            "n_canvas": len(imgs),
            "_bitcost_scored": None,  # filled lazily by run_select_mode
        }

    def run_reweight(
        self,
        video: str,
        question: str,
        num_frames: int = 32,
        max_pixels: int = 150000,
        max_new_tokens: int = 64,
        engine: str = "hevc",
        budget_k: int | None = None,
        keep_ratio: float = 0.75,
        qtype: QuestionType | None = None,
    ) -> ForwardStats:
        """Legacy canvas-level reweight (full Video-MME row B path)."""
        qtype = qtype or classify_question(question)
        t0 = time.perf_counter()
        enc = self.encode_codec(
            video, num_frames=num_frames, max_pixels=max_pixels, engine=engine
        )
        rw = reweight_codec_canvases(
            enc["images"],
            enc["src_positions"],
            qtype=qtype,
            budget_k=budget_k,
            keep_ratio=keep_ratio,
        )
        inputs, n_vis = self._build_inputs_from_codec_payload(
            question, rw.images, rw.src_positions, enc["fps"], max_pixels
        )
        answer = self._generate(inputs, max_new_tokens)
        wall = time.perf_counter() - t0
        return ForwardStats(
            mode="reweight",
            question=question,
            qtype=qtype.value,
            n_vis_tokens=n_vis,
            n_patches=rw.n_after,
            wall_s=wall,
            answer=answer,
            reweight=rw,
            extra={
                "n_canvas_before": enc["n_canvas"],
                "n_canvas_after": len(rw.images),
                "mean_motion_all": rw.mean_motion_all,
                "mean_motion_kept": rw.mean_motion_kept,
                "kept_canvas_indices": rw.kept_canvas_indices,
                "alpha": rw.alpha,
                "beta": rw.beta,
                "budget_k": rw.budget_k,
            },
        )

    def run_select_mode(
        self,
        question: str,
        enc: dict[str, Any],
        *,
        mode: SelectMode,
        budget_k: int,
        max_pixels: int = 150000,
        max_new_tokens: int = 64,
        keep_ratio: float = 0.75,
        qtype: QuestionType | None = None,
        seed: int = 0,
    ) -> ForwardStats:
        """Patch-level same-K select using a precomputed codec payload."""
        qtype = qtype or classify_question(question)
        t0 = time.perf_counter()
        score_source = "proxy"
        if mode == "queryip_bitcost":
            from .mvres_score import score_patches_bitcost, select_patches_bitcost

            video = enc.get("video")
            if not video:
                raise ValueError(
                    "enc['video'] required for queryip_bitcost "
                    "(re-run encode_codec after dual-mode update)"
                )
            if enc.get("_bitcost_scored") is None:
                enc["_bitcost_scored"] = score_patches_bitcost(
                    str(video),
                    enc["images"],
                    enc["src_positions"],
                    meta=enc.get("meta") or {},
                )
            scored = enc["_bitcost_scored"]
            rw = select_patches_bitcost(
                enc["images"],
                enc["src_positions"],
                scored,
                qtype=qtype,
                budget_k=budget_k,
                keep_ratio=keep_ratio,
                proxy_fallback=True,
            )
            score_source = str(
                (rw.knobs or {}).get("score_source")
                or scored.get("score_source")
                or "bitcost"
            )
        else:
            rw = select_patches_samek(
                enc["images"],
                enc["src_positions"],
                mode=mode,
                qtype=qtype,
                budget_k=budget_k,
                keep_ratio=keep_ratio,
                seed=seed,
            )
            if mode == "queryip":
                score_source = "proxy"
            else:
                score_source = "none"
        inputs, n_vis = self._build_inputs_from_codec_payload(
            question, rw.images, rw.src_positions, enc["fps"], max_pixels
        )
        answer = self._generate(inputs, max_new_tokens)
        wall = time.perf_counter() - t0
        return ForwardStats(
            mode=f"samek_{mode}",
            question=question,
            qtype=qtype.value,
            n_vis_tokens=n_vis,
            n_patches=rw.n_after,
            wall_s=wall,
            answer=answer,
            reweight=rw,
            extra={
                "select_mode": mode,
                "score_source": score_source,
                "n_canvas_before": enc["n_canvas"],
                "n_canvas_after": len(rw.images),
                "n_vis_tokens": n_vis,
                "mean_motion_all": rw.mean_motion_all,
                "mean_motion_kept": rw.mean_motion_kept,
                "kept_patch_indices": rw.kept_patch_indices,
                "kept_canvas_indices": rw.kept_canvas_indices,
                "alpha": rw.alpha,
                "beta": rw.beta,
                "budget_k": rw.budget_k,
                "seed": seed,
            },
        )
