#!/usr/bin/env python3
"""Diagnose VGGT vs COLMAP sparse-depth median scale (no training / no loss change).

For each frame: median_scale = median(gt / pred) on valid COLMAP pixels,
then L1 after aligning pred *= scale. Compares base model.pt vs optional
depth_head-only overlays.

Example:
  python3 tools/diag_vggt_colmap_scale.py \\
    --data_dir data/mip360/kitchen \\
    --resume ../vggt/models/model.pt \\
    --depth_heads ../vggt/models/depth_head_ft_mip360.pt \\
    --use_mask --max_frames 40 \\
    --out_csv ../../3DGS/eval_results/depthft_scale_diag_kitchen.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_GO_DIR = os.path.dirname(_TOOLS_DIR)
_REPO_3DGS = os.path.dirname(_GO_DIR)
_VGGT_DIR = os.path.join(_REPO_3DGS, "vggt")
sys.path.insert(0, _TOOLS_DIR)
sys.path.insert(0, _VGGT_DIR)

from colmap_sparse_depth import (  # noqa: E402
    iter_scene_sparse_depths,
    load_exclude_indices,
)
from finetune_depth_head_colmap import (  # noqa: E402
    ColmapSparseDepthDataset,
    extract_depth_bhw,
    forward_tokens,
)
from vggt_depthft_ckpt import load_vggt_with_optional_depth_head  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402


def median_scale_and_l1(
    pred: torch.Tensor, gt: torch.Tensor, min_valid: int = 50
) -> Optional[Tuple[float, float, int]]:
    """Returns (median_scale, aligned_l1, n_valid) or None if too few pixels."""
    mask = gt > 0
    n = int(mask.sum().item())
    if n < min_valid:
        return None
    pred_m = pred[mask]
    gt_m = gt[mask]
    ratio = gt_m / pred_m.clamp(min=1e-6)
    scale = float(ratio.median().item())
    aligned = pred * scale
    l1 = float((aligned[mask] - gt_m).abs().mean().item())
    return scale, l1, n


@torch.no_grad()
def predict_depth(model: VGGT, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """image: CHW float [0,1] → HxW depth on CPU."""
    # VGGT expects [B, S, C, H, W]
    images = image.unsqueeze(0).unsqueeze(1).to(device)
    tokens, patch_start_idx = forward_tokens(model, images)
    depth, _conf = model.depth_head(
        tokens, images=images, patch_start_idx=patch_start_idx
    )
    d = extract_depth_bhw(depth)[0].float().cpu()
    return d


def load_model(
    resume: str,
    depth_head: Optional[str],
    device: torch.device,
) -> VGGT:
    model = VGGT()
    if depth_head:
        mode, base_used = load_vggt_with_optional_depth_head(
            model,
            depth_head,
            base_model_path=resume,
            map_location="cpu",
        )
        print(f"[model] mode={mode} base={base_used} head={depth_head}")
    else:
        state = torch.load(resume, map_location="cpu")
        if isinstance(state, dict) and "model" in state and not any(
            k.startswith("aggregator") or k.startswith("depth_head") for k in state
        ):
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] missing keys: {len(missing)}")
        if unexpected:
            print(f"[warn] unexpected keys: {len(unexpected)}")
        print(f"[model] full base only: {resume}")
    model.to(device)
    model.eval()
    return model


def summarize(scales: List[float], l1s: List[float], tag: str) -> None:
    if not scales:
        print(f"[summary] {tag}: no valid frames")
        return
    s = np.asarray(scales, dtype=np.float64)
    l = np.asarray(l1s, dtype=np.float64)
    print(
        f"[summary] {tag}: n={len(s)} "
        f"scale median={np.median(s):.4f} mean={s.mean():.4f} "
        f"std={s.std():.4f} min={s.min():.4f} max={s.max():.4f} | "
        f"aligned_L1 median={np.median(l):.4f} mean={l.mean():.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose median scale between VGGT depth and COLMAP sparse depth"
    )
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--resume", type=str, required=True, help="Full VGGT model.pt")
    p.add_argument(
        "--depth_heads",
        type=str,
        nargs="*",
        default=[],
        help="Optional depth_head-only overlays to compare (plus base)",
    )
    p.add_argument("--use_mask", action="store_true")
    p.add_argument("--exclude_sparse", type=str, default=None)
    p.add_argument("--max_side", type=int, default=518)
    p.add_argument(
        "--min_valid",
        type=int,
        default=200,
        help="Dataset filter on COLMAP n_valid before resize",
    )
    p.add_argument(
        "--stats_min_valid",
        type=int,
        default=10,
        help="Min valid pixels AFTER resize to count a frame in scale stats",
    )
    p.add_argument("--max_frames", type=int, default=40)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Per-frame CSV (e.g. 3DGS/eval_results/depthft_scale_diag_kitchen.csv)",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    exclude = load_exclude_indices(args.exclude_sparse)
    frames = iter_scene_sparse_depths(
        args.data_dir,
        exclude_indices=exclude,
        use_mask=args.use_mask,
    )
    dataset = ColmapSparseDepthDataset(
        frames,
        max_side=args.max_side,
        min_valid=args.min_valid,
    )
    n = min(len(dataset), args.max_frames) if args.max_frames > 0 else len(dataset)
    # Peek post-resize density (nearest resize used to destroy sparse hits)
    peek_n = [int((dataset[i]["depth"] > 0).sum().item()) for i in range(min(5, n))]
    print(
        f"[data] {args.data_dir} usable={len(dataset)} eval_frames={n} "
        f"use_mask={args.use_mask} max_side={args.max_side} "
        f"stats_min_valid={args.stats_min_valid}"
    )
    print(f"[data] post-resize n_valid (first {len(peek_n)} frames): {peek_n}")

    # tag -> list of overlays; empty string = base only
    configs: List[Tuple[str, Optional[str]]] = [("base", None)]
    for hp in args.depth_heads:
        tag = Path(hp).stem
        configs.append((tag, hp))

    rows: List[Dict[str, object]] = []
    summaries: Dict[str, Tuple[List[float], List[float]]] = {}

    for tag, head_path in configs:
        print(f"========== {tag} ==========")
        model = load_model(args.resume, head_path, device)
        scales: List[float] = []
        l1s: List[float] = []
        n_skip_low = 0
        for i in tqdm(range(n), desc=tag):
            sample = dataset[i]
            image = sample["image"]
            gt = sample["depth"]
            pred = predict_depth(model, image, device)
            if pred.shape != gt.shape:
                pred = F.interpolate(
                    pred.unsqueeze(0).unsqueeze(0),
                    size=gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            stats = median_scale_and_l1(pred, gt, min_valid=args.stats_min_valid)
            if stats is None:
                n_skip_low += 1
                continue
            scale, l1, n_valid = stats
            scales.append(scale)
            l1s.append(l1)
            rows.append(
                {
                    "tag": tag,
                    "depth_head": head_path or "",
                    "frame_index": sample["index"],
                    "name": sample["name"],
                    "n_valid": n_valid,
                    "median_scale": f"{scale:.6f}",
                    "aligned_l1": f"{l1:.6f}",
                }
            )
        summaries[tag] = (scales, l1s)
        if n_skip_low:
            print(
                f"[warn] {tag}: skipped {n_skip_low}/{n} frames "
                f"(post-resize n_valid < {args.stats_min_valid})"
            )
        summarize(scales, l1s, tag)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "tag",
                    "depth_head",
                    "frame_index",
                    "name",
                    "n_valid",
                    "median_scale",
                    "aligned_l1",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] wrote {out} ({len(rows)} rows)")

    print("========== compare ==========")
    for tag, (scales, l1s) in summaries.items():
        summarize(scales, l1s, tag)
    print(
        "[note] Extreme median_scale (<<0.1 or >>10) or huge std → stop and ask "
        "before changing loss. Mild scale≈1 with similar L1 is expected if "
        "finetune already used median_scale_l1."
    )


if __name__ == "__main__":
    main()
