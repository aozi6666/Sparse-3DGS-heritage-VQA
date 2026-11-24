# Mage-VL VLMEvalKit Adapter Design

**Date:** 2026-09-29  
**Status:** Approved / Implemented  
**Goal:** Register Mage-VL in VLMEvalKit for video-benchmark eval (VideoMME / MVBench / LongVideoBench / VSI-Bench).

## Decisions (approved)

- Thin wrapper (single file), not a Qwen-style package split.
- Video path: `video_backend="codec"` + HEVC (`engine: "hevc"`).
- No streaming / System-1 gate (offline QA eval only).
- Default canvas budget: `nframe=32` (paper `tc-32`).

## Scope

### In

1. `VLMEvalKit/vlmeval/vlm/mage_vl.py` — class `MageVLChat(BaseModel)`
2. Import in `vlmeval/vlm/__init__.py`
3. Register `"Mage-VL"` in `vlmeval/config.py` via `partial(vlm.MageVLChat, ...)`

### Out

- Streaming gate / OVO-Bench / SoccerNet timing protocols
- Custom per-dataset prompts (use VLMEvalKit defaults)
- vLLM / SGLang online path
- Frames backend (can be added later as optional kwargs)

## Class contract

| Flag / param | Value |
|---|---|
| `VIDEO_LLM` | `True` |
| `INTERLEAVE` | `True` |
| `model_path` | `/root/autodl-tmp/Mage/mage_vl/models` (overrideable) |
| `nframe` | `32` → `codec_config.target_canvas` |
| `max_pixels` | `150000` |
| `max_new_tokens` | `256` (match Mage CLI default; overrideable) |
| `codec_engine` | `"traditional"` → HEVC |
| generation | `do_sample=False` |

### `generate_inner(message, dataset=None)`

1. Parse VLMEvalKit message list `[{type, value}, ...]` into images / one video path / text question.
2. Build chat messages like Mage `inference.py` offline path (`type: image|video` + text).
3. `processor.apply_chat_template` → `processor(...)` with codec kwargs for video, or `images=` for stills.
4. `model.generate` → decode new tokens → return stripped string.

## Usage

```bash
cd VLMEvalKit
python run.py --data VideoMME MVBench LongVideoBench --model Mage-VL --verbose
```

## Prerequisite / risk

Local `Mage/mage_vl/models/` currently has weights + partial remote-code files. Full HF snapshot (`modeling_mage_vl.py`, `processing_mage_vl.py`, tokenizer, etc.) is required for `trust_remote_code=True` load. Adapter assumes a complete model dir or HF id `microsoft/Mage-VL`.

## Success criteria

- `from vlmeval.config import supported_VLM; supported_VLM['Mage-VL']` resolves.
- One video sample through `generate_inner` returns non-empty text without crashing (once model files are complete).
