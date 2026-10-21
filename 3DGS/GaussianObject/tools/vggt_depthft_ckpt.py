"""VGGT depth_head-only checkpoint I/O (disk-saving finetune artifacts).

Format:
  {
    "format": "vggt_depth_head_only",
    "version": 1,
    "base_model": "<optional path hint to full model.pt>",
    "depth_head": { "depth_head.*": tensor, ... },
  }

Load: full base model.pt, then overlay depth_head (strict=False).
Numerically equivalent to saving/loading a full finetuned state_dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

DEPTH_HEAD_ONLY_FORMAT = "vggt_depth_head_only"
DEPTH_HEAD_ONLY_VERSION = 1


def is_depth_head_only_ckpt(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("format") == DEPTH_HEAD_ONLY_FORMAT


def extract_depth_head_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k.startswith("depth_head.")
    }


def save_depth_head_only(
    model: nn.Module,
    out_path: str | Path,
    base_model: Optional[str] = None,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": DEPTH_HEAD_ONLY_FORMAT,
        "version": DEPTH_HEAD_ONLY_VERSION,
        "base_model": str(base_model) if base_model else "",
        "depth_head": extract_depth_head_state(model),
    }
    torch.save(payload, out)
    return out.resolve()


def _unwrap_full_state(state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith("aggregator") or k.startswith("depth_head") for k in state
    ):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict dict, got {type(state)}")
    return state


def normalize_depth_head_dict(depth_head: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Accept keys with or without 'depth_head.' prefix."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in depth_head.items():
        key = k if k.startswith("depth_head.") else f"depth_head.{k}"
        out[key] = v
    return out


def load_vggt_with_optional_depth_head(
    model: nn.Module,
    model_path: str,
    *,
    base_model_path: Optional[str] = None,
    map_location: Any = "cpu",
) -> Tuple[str, Optional[str]]:
    """Load weights into ``model``.

    - If ``model_path`` is a full VGGT state_dict → load as today.
    - If ``model_path`` is depth_head-only → load ``base_model_path`` (or hint /
      default beside repo) then overlay depth_head.

    Returns (mode, base_used) where mode is "full" or "depth_head_only".
    """
    ckpt = torch.load(model_path, map_location=map_location)

    if is_depth_head_only_ckpt(ckpt):
        base = base_model_path or (ckpt.get("base_model") or "").strip() or None
        if not base or not Path(base).is_file():
            # Fall back: sibling model.pt next to the depth-head file
            sibling = Path(model_path).resolve().parent / "model.pt"
            if sibling.is_file():
                base = str(sibling)
            else:
                raise FileNotFoundError(
                    "depth_head-only checkpoint requires a full base model.pt; "
                    f"set --base_model_path (tried hint={ckpt.get('base_model')!r}, "
                    f"sibling={sibling})"
                )
        base_state = _unwrap_full_state(torch.load(base, map_location=map_location))
        missing, unexpected = model.load_state_dict(base_state, strict=False)
        if missing:
            print(f"[ckpt] base missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[ckpt] base unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

        dh = normalize_depth_head_dict(ckpt["depth_head"])
        missing_dh, unexpected_dh = model.load_state_dict(dh, strict=False)
        # Only depth_head keys should apply; unexpected empty; missing = all non-head
        n_applied = len(dh) - len(unexpected_dh)
        print(
            f"[ckpt] depth_head-only: base={base}  "
            f"overlay_keys={len(dh)} applied≈{n_applied} unexpected={len(unexpected_dh)}"
        )
        if unexpected_dh:
            print(f"[ckpt] depth_head unexpected (showing up to 5): {unexpected_dh[:5]}")
        return "depth_head_only", base

    state = _unwrap_full_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    return "full", None
