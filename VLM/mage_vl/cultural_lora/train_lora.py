#!/usr/bin/env python3
"""Cultural LoRA smoke/full trainer: freeze Mage-ViT, PEFT on LLM, adapter-only saves."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import (
    MixedCulturalDataset,
    build_chat_messages,
    sample_images,
    subsample_pil_list,
)
from paths import DEFAULT_CIQI_DIR, DEFAULT_MODEL_DIR, DEFAULT_MSVD_DIR, DEFAULT_OUTPUT_DIR


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--ciqi-dir", type=Path, default=DEFAULT_CIQI_DIR)
    ap.add_argument("--msvd-dir", type=Path, default=DEFAULT_MSVD_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "smoke")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=150_000,
        help="Per-sample visual pixel budget (plan default ~150k)",
    )
    ap.add_argument(
        "--max-answer-chars",
        type=int,
        default=1024,
        help="Cap assistant text chars (CiQi essays); MSVD answers are short and unaffected",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=3072,
        help="Hard cap on tokenized sequence length (truncate from the end)",
    )
    ap.add_argument(
        "--ciqi-max-images",
        type=int,
        default=2,
        help="Max images per CiQi sample (2 keeps multi-view; set 1 if still OOM)",
    )
    ap.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more VRAM)",
    )
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ciqi-only", action="store_true")
    ap.add_argument("--msvd-only", action="store_true")
    ap.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 (default on)",
    )
    ap.add_argument("--fp16", action="store_true", help="Use float16 instead of bf16")
    return ap.parse_args()


def _set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_local_model(model_dir: Path) -> Path:
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model dir missing: {model_dir}")
    index = model_dir / "model.safetensors.index.json"
    if not index.is_file() and not any(model_dir.glob("*.safetensors")):
        raise SystemExit(f"no local weights under {model_dir}; refuse HF download")
    return model_dir


def _freeze_visual(model) -> None:
    visual = model.visual
    for p in visual.parameters():
        p.requires_grad = False
    visual.eval()


def _enable_memory_saver(model, enabled: bool = True) -> None:
    """Gradient checkpointing + disable KV cache (no quality change, less activation VRAM)."""
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if not enabled:
        return

    # Prefer HF API on the outer causal LM
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    lm = model.model.language_model
    # PEFT needs input grads when base is frozen
    if hasattr(lm, "enable_input_require_grads"):
        lm.enable_input_require_grads()
    elif hasattr(lm, "get_base_model"):
        base = lm.get_base_model()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        if hasattr(base, "gradient_checkpointing_enable"):
            try:
                base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                base.gradient_checkpointing_enable()

    print("[train] gradient_checkpointing=ON, use_cache=False")


def _attach_lora(model, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    # Freeze all then LoRA-unfreeze LLM linears only
    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        # language_model is the inner Qwen stack (not ForCausalLM)
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    # Wrap inner language model only (Qwen3 stack)
    lm = model.model.language_model
    model.model.language_model = get_peft_model(lm, lora_config)
    _freeze_visual(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    return model


def _save_adapter_only(model, output_dir: Path, tag: str, keep_last: int = 1) -> Path:
    """Save PEFT adapter weights only; rotate so at most `keep_last` checkpoint-* dirs remain."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / tag
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    peft_model = model.model.language_model
    peft_model.save_pretrained(ckpt_dir)

    # Guard: never write full base shards into output
    forbidden = list(ckpt_dir.glob("model-*-of-*.safetensors")) + list(
        ckpt_dir.glob("pytorch_model*.bin")
    )
    if forbidden:
        raise RuntimeError(f"full weights leaked into {ckpt_dir}: {forbidden}")

    # Rotate older checkpoint-* dirs only (adapter/ / final handled by caller)
    if tag.startswith("checkpoint-"):
        ckpts = sorted(
            [
                p
                for p in output_dir.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint-")
            ],
            key=lambda p: p.stat().st_mtime,
        )
        while len(ckpts) > keep_last:
            old = ckpts.pop(0)
            print(f"[train] prune old ckpt {old.name}")
            shutil.rmtree(old, ignore_errors=True)

    meta = {
        "format": "peft_adapter_only",
        "base_model": str(DEFAULT_MODEL_DIR),
        "tag": tag,
    }
    (ckpt_dir / "cultural_lora_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[train] saved adapter -> {ckpt_dir}")
    return ckpt_dir


def _process_pair(processor, prompt_text: str, full_text: str, images, max_pixels: int):
    try:
        full_inputs = processor(
            text=[full_text],
            images=images,
            return_tensors="pt",
            max_pixels=max_pixels,
        )
        prompt_inputs = processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
            max_pixels=max_pixels,
        )
    except TypeError:
        full_inputs = processor(text=[full_text], images=images, return_tensors="pt")
        prompt_inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
    return full_inputs, prompt_inputs


def _pack_batch(full_inputs, prompt_inputs, device, dtype) -> dict:
    input_ids = full_inputs["input_ids"].to(device)
    attention_mask = full_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    labels = input_ids.clone()
    prompt_len = min(int(prompt_inputs["input_ids"].shape[1]), int(input_ids.shape[1]))
    labels[:, :prompt_len] = -100

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "use_cache": False,
    }
    for key in ("pixel_values", "image_grid_thw", "patch_positions"):
        if key in full_inputs and full_inputs[key] is not None:
            val = full_inputs[key]
            if hasattr(val, "to"):
                if key == "pixel_values":
                    val = val.to(device=device, dtype=dtype)
                else:
                    val = val.to(device)
            batch[key] = val
    return batch


def _batch_ok(batch: dict, max_seq_len: int) -> bool:
    """Require: fits length budget AND has at least one supervised (answer) token."""
    if (batch["labels"] != -100).sum().item() == 0:
        return False
    if max_seq_len > 0 and batch["input_ids"].shape[1] > max_seq_len:
        return False
    return True


def _build_batch(
    processor,
    sample,
    device,
    dtype,
    max_pixels: int,
    max_answer_chars: int,
    max_seq_len: int,
    ciqi_max_images: int,
):
    """
    Fit sample under max_seq_len **without cutting the answer**.

    If multimodal prompt is too long, uniformly drop frames/images then lower
    max_pixels and re-encode. Truncating the tail would wipe short MSVD answers
    after many video tokens — that path is intentionally removed.
    """
    images = sample_images(sample, ciqi_max_images=ciqi_max_images)
    cur_pixels = int(max_pixels)
    min_pixels = 48_000

    for _attempt in range(24):
        messages = build_chat_messages(
            sample,
            max_answer_chars=max_answer_chars,
            ciqi_max_images=ciqi_max_images,
            num_images=len(images),
        )
        prompt_text = processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_inputs, prompt_inputs = _process_pair(
            processor, prompt_text, full_text, images, cur_pixels
        )
        batch = _pack_batch(full_inputs, prompt_inputs, device, dtype)
        if _batch_ok(batch, max_seq_len):
            return batch

        # Prefer keeping the full answer: shrink visual budget instead.
        if len(images) > 1:
            images = subsample_pil_list(images, max(1, len(images) // 2))
            continue
        if cur_pixels > min_pixels:
            cur_pixels = max(min_pixels, int(cur_pixels * 0.7))
            continue
        break

    # Last try already packed; if still bad, raise so caller can skip with context
    if not _batch_ok(batch, max_seq_len):
        raise RuntimeError(
            f"cannot fit sample with supervised answer "
            f"(seq={batch['input_ids'].shape[1]} max={max_seq_len} "
            f"frames={len(images)} pixels={cur_pixels})"
        )
    return batch


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model_dir = _assert_local_model(args.model)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_ciqi = not args.msvd_only
    include_msvd = not args.ciqi_only
    dataset = MixedCulturalDataset(
        ciqi_dir=args.ciqi_dir,
        msvd_dir=args.msvd_dir,
        max_frames=args.max_frames,
        include_ciqi=include_ciqi,
        include_msvd=include_msvd,
    )
    print(f"[train] dataset size={len(dataset)} ciqi={include_ciqi} msvd={include_msvd}")
    print(
        f"[train] mem knobs: max_pixels={args.max_pixels} max_answer_chars={args.max_answer_chars} "
        f"max_seq_len={args.max_seq_len} ciqi_max_images={args.ciqi_max_images} "
        f"grad_ckpt={not args.no_gradient_checkpointing}"
    )
    if len(dataset) == 0:
        raise SystemExit("empty dataset")

    loader = DataLoader(
        dataset,
        batch_size=1,  # variable-length multimodal; collate manually
        shuffle=True,
        num_workers=0,
        collate_fn=lambda xs: xs[0],
    )

    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype = torch.float16 if args.fp16 else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] loading processor/model from {model_dir} dtype={dtype} device={device}")

    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model = _attach_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    _enable_memory_saver(model, enabled=not args.no_gradient_checkpointing)
    model.to(device)
    model.train()
    # visual stays eval / frozen
    model.visual.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)

    step = 0
    optim.zero_grad(set_to_none=True)
    running = 0.0
    oom_skips = 0
    t0 = time.time()
    data_iter = iter(loader)

    pbar = tqdm(total=args.max_steps, desc="cultural-lora")
    while step < args.max_steps:
        try:
            sample = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            sample = next(data_iter)

        try:
            batch = _build_batch(
                processor,
                sample,
                device=device,
                dtype=dtype,
                max_pixels=args.max_pixels,
                max_answer_chars=args.max_answer_chars,
                max_seq_len=args.max_seq_len,
                ciqi_max_images=args.ciqi_max_images,
            )
        except Exception as e:
            print(f"[train] skip sample ({sample.get('id')}): {e}")
            continue

        try:
            out = model(**batch)
            loss = out.loss
            if loss is None:
                print("[train] skip: model returned no loss")
                continue
            (loss / args.grad_accum).backward()
            running += float(loss.detach().item())
        except torch.cuda.OutOfMemoryError:
            oom_skips += 1
            optim.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[train] OOM skip sample ({sample.get('id')}); "
                f"oom_skips={oom_skips} seq_len={batch['input_ids'].shape[1]}"
            )
            continue

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

        step += 1
        pbar.update(1)
        if step % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            pbar.set_postfix(loss=f"{avg:.4f}", t=f"{time.time() - t0:.0f}s", oom=oom_skips)

        if args.save_every > 0 and step % args.save_every == 0:
            _save_adapter_only(model, output_dir, tag=f"checkpoint-{step}", keep_last=1)

    pbar.close()
    # Flush leftover grads
    if step % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

    final = _save_adapter_only(model, output_dir, tag="final", keep_last=1)
    # Canonical adapter/ only; drop intermediate checkpoint-* and duplicate final/
    canonical = output_dir / "adapter"
    if canonical.exists():
        shutil.rmtree(canonical)
    shutil.copytree(final, canonical)
    shutil.rmtree(final, ignore_errors=True)
    for p in list(output_dir.glob("checkpoint-*")):
        shutil.rmtree(p, ignore_errors=True)
        print(f"[train] pruned {p.name} (keep adapter/ only)")
    print(f"[train] done. adapter at {canonical} (oom_skips={oom_skips})")

    # Write run config (no weights)
    (output_dir / "train_args.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
