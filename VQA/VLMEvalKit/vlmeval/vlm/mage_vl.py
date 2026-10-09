"""Mage-VL adapter for VLMEvalKit (offline codec + HEVC video path)."""

from __future__ import annotations

import os
import warnings

import torch
from PIL import Image

from .base import BaseModel

DEFAULT_MODEL_PATH = "/root/autodl-tmp/Mage/mage_vl/models"

# Files required for trust_remote_code local load (weights may be sharded).
REQUIRED_MODEL_FILES = (
    "config.json",
    "modeling_mage_vl.py",
    "processing_mage_vl.py",
    "video_processing_mage_vl.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
)


def _assert_mage_model_complete(model_path: str) -> None:
    """Raise a clear error if the local Mage-VL directory is incomplete."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Mage-VL model_path is not a directory: {model_path}. "
            "Copy a full microsoft/Mage-VL snapshot into this path."
        )
    missing = [name for name in REQUIRED_MODEL_FILES if not os.path.exists(os.path.join(model_path, name))]
    has_index = os.path.exists(os.path.join(model_path, "model.safetensors.index.json"))
    has_shards = any(
        name.startswith("model-") and name.endswith(".safetensors")
        for name in os.listdir(model_path)
    )
    has_single = os.path.exists(os.path.join(model_path, "model.safetensors"))
    if not (has_index or has_shards or has_single):
        missing.append("model.safetensors.index.json (or model-*-of-*.safetensors / model.safetensors)")
    elif has_shards and not has_index:
        missing.append("model.safetensors.index.json")
    if missing:
        raise FileNotFoundError(
            "Mage-VL local checkpoint is incomplete under "
            f"{model_path}. Missing: {', '.join(missing)}. "
            "Import the missing files from microsoft/Mage-VL before running eval."
        )


class MageVLChat(BaseModel):
    """Thin wrapper around Mage-VL offline inference (codec-native HEVC)."""

    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        nframe: int = 32,
        max_pixels: int = 150000,
        max_new_tokens: int = 256,
        codec_engine: str = "traditional",
        **kwargs,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoProcessor

        assert model_path is not None
        self.model_path = model_path
        self.nframe = nframe
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.codec_engine = codec_engine

        if kwargs:
            warnings.warn(f"Unused MageVLChat kwargs ignored: {sorted(kwargs.keys())}")

        _assert_mage_model_complete(self.model_path)

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        ).eval()
        torch.cuda.empty_cache()

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
                raise ValueError(f"Unsupported message type for Mage-VL: {typ}")
        question = "\n".join(texts).strip()
        if not question:
            question = "Describe this media."
        return images, videos, question

    def generate_inner(self, message, dataset=None):
        images, videos, question = self._parse_message(message)
        if videos and images:
            raise ValueError("Mage-VL adapter expects either images or one video, not both.")
        if len(videos) > 1:
            raise ValueError("Mage-VL adapter supports at most one video per sample.")

        if videos:
            media_type = "video"
            video_path = videos[0]
        elif images:
            media_type = "image"
        else:
            raise ValueError("Mage-VL adapter requires at least one image or video.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": media_type},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if media_type == "image":
            pil_images = [Image.open(p).convert("RGB") for p in images]
            inputs = self.processor(
                text=[text], images=pil_images, return_tensors="pt"
            )
        else:
            # Plan: traditional codec_engine maps to HEVC.
            codec_config = {
                "engine": "hevc" if self.codec_engine == "traditional" else "dcvc-rt",
                "target_canvas": self.nframe,
                "patch": 16,
            }
            if self.codec_engine == "neural":
                codec_config["dcvc"] = {
                    "pkg_dir": os.path.join(self.model_path, "neural_codec"),
                    "device": str(self.model.device),
                }
            inputs = self.processor(
                text=[text],
                videos=[video_path],
                video_backend="codec",
                max_pixels=self.max_pixels,
                codec_config=codec_config,
                return_tensors="pt",
                padding=True,
            )

        inputs = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        answer = self.processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return answer.strip()
