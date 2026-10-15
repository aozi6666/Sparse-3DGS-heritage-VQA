#!/usr/bin/env python3
"""
Decoder-only VGGT depth_head finetune on COLMAP sparse depth.

- Loads resume weights read-only; refuses to overwrite model.pt
- Freezes aggregator / camera / point / track; trains depth_head only
- Saves depth_head-only checkpoint (not a full VGGT copy)
- Single scene: --data_dir (+ optional --exclude_sparse)
- Multi scene: --data_dirs a b c  (or comma-separated); frames concatenated
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_GO_DIR = os.path.dirname(_TOOLS_DIR)
_REPO_3DGS = os.path.dirname(_GO_DIR)
_VGGT_DIR = os.path.join(_REPO_3DGS, "vggt")
sys.path.insert(0, _TOOLS_DIR)
sys.path.insert(0, _VGGT_DIR)

from colmap_sparse_depth import (  # noqa: E402
    FrameSparseDepth,
    iter_scene_sparse_depths,
    load_exclude_indices,
    resize_sparse_depth,
)
from vggt_depthft_ckpt import save_depth_head_only  # noqa: E402

from vggt.models.vggt import VGGT  # noqa: E402


def assert_safe_checkpoint_out(out_path: str, resume_path: str) -> Path:
    out = Path(out_path).resolve()
    resume = Path(resume_path).resolve()
    if out == resume:
        print(f"[ERR] refusing to overwrite resume checkpoint: {out}", file=sys.stderr)
        sys.exit(1)
    if out.name == "model.pt":
        print(
            "[ERR] refusing to write to a file named model.pt "
            "(keep the original VGGT weights intact)",
            file=sys.stderr,
        )
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def pad_to_patch_multiple(
    image: torch.Tensor, depth: torch.Tensor, patch_size: int = 14
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match vggt_visual_hull_enhanced.adjust_image_size_for_vggt (bilinear / nearest)."""
    _, h, w = image.shape
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    if new_h == h and new_w == w:
        return image, depth
    image = F.interpolate(
        image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    depth = F.interpolate(
        depth.unsqueeze(0).unsqueeze(0),
        size=(new_h, new_w),
        mode="nearest",
    ).squeeze(0).squeeze(0)
    return image, depth


def load_rgb_tensor(path: str, max_side: int):
    """RGB float tensor in [0, 1], CHW; optionally downscale by max side.

    Returns (image_chw, scale, (orig_h, orig_w)).
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 1.0
    if max_side > 0 and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), scale, (h, w)


class ColmapSparseDepthDataset(Dataset):
    def __init__(
        self,
        frames: List[FrameSparseDepth],
        max_side: int = 518,
        min_valid: int = 200,
        patch_size: int = 14,
    ):
        self.frames = [f for f in frames if (not f.excluded) and f.n_valid >= min_valid and f.image_path]
        self.max_side = max_side
        self.patch_size = patch_size
        if not self.frames:
            raise RuntimeError(
                "No training frames left after exclude / min_valid / missing image filters"
            )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame = self.frames[idx]
        image, scale, (orig_h, orig_w) = load_rgb_tensor(frame.image_path, self.max_side)
        depth = frame.depth
        if scale != 1.0:
            nh, nw = image.shape[1], image.shape[2]
            depth = resize_sparse_depth(depth, nh, nw)
        else:
            # image may still match original size
            if depth.shape[0] != image.shape[1] or depth.shape[1] != image.shape[2]:
                depth = resize_sparse_depth(depth, image.shape[1], image.shape[2])
        depth_t = torch.from_numpy(depth.astype(np.float32))
        image, depth_t = pad_to_patch_multiple(image, depth_t, self.patch_size)
        return {
            "image": image,
            "depth": depth_t,
            "name": frame.name,
            "index": frame.index,
            "n_valid": int((depth_t > 0).sum().item()),
        }


def freeze_all_but_depth_head(model: VGGT) -> None:
    for name, param in model.named_parameters():
        if name.startswith("depth_head."):
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.aggregator.eval()
    if model.camera_head is not None:
        model.camera_head.eval()
    if model.point_head is not None:
        model.point_head.eval()
    if model.track_head is not None:
        model.track_head.eval()
    model.depth_head.train()


def median_scale_l1(
    pred: torch.Tensor, gt: torch.Tensor, min_valid: int = 50
) -> Optional[torch.Tensor]:
    """pred/gt: HxW; align pred by median(gt/pred) on valid pixels, then L1."""
    mask = gt > 0
    n = int(mask.sum().item())
    if n < min_valid:
        return None
    pred_m = pred[mask]
    gt_m = gt[mask]
    ratio = gt_m / pred_m.clamp(min=1e-6)
    scale = ratio.median()
    aligned = pred * scale
    return (aligned[mask] - gt_m).abs().mean()


def extract_depth_bhw(pred_depth: torch.Tensor) -> torch.Tensor:
    """Normalize VGGT depth output to [B, H, W]."""
    d = pred_depth
    if d.ndim == 5:  # B, S, H, W, 1
        d = d[:, 0, :, :, 0]
    elif d.ndim == 4:  # B, S, H, W  or  B, H, W, 1
        if d.shape[-1] == 1:
            d = d[..., 0]
        else:
            d = d[:, 0]
    elif d.ndim == 3:  # H, W[, 1] without batch — add batch
        if d.shape[-1] == 1:
            d = d[..., 0]
        d = d.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected depth shape: {tuple(pred_depth.shape)}")
    return d


@torch.no_grad()
def forward_tokens(model: VGGT, images: torch.Tensor):
    return model.aggregator(images)


def train_one_epoch(
    model: VGGT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    min_valid: int,
    grad_clip: float,
) -> float:
    total = 0.0
    n_steps = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        image = batch["image"].to(device)  # B,C,H,W
        gt = batch["depth"].to(device)  # B,H,W
        # VGGT expects [B,S,C,H,W]
        images = image.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                tokens, patch_start_idx = forward_tokens(model, images)
            depth, _conf = model.depth_head(
                tokens, images=images, patch_start_idx=patch_start_idx
            )
            pred = extract_depth_bhw(depth)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(
                    pred.unsqueeze(1),
                    size=gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            losses = []
            for b in range(gt.shape[0]):
                loss_b = median_scale_l1(pred[b], gt[b], min_valid)
                if loss_b is not None:
                    losses.append(loss_b)
            if not losses:
                continue
            loss = torch.stack(losses).mean()

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.depth_head.parameters() if p.requires_grad),
                grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.detach().item())
        n_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        del tokens, depth, pred, images, image, gt, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total / max(n_steps, 1)


def resolve_data_dirs(data_dir: Optional[str], data_dirs: Optional[List[str]]) -> List[str]:
    """Collect scene roots from --data_dir and/or --data_dirs (comma ok in each token)."""
    dirs: List[str] = []
    if data_dirs:
        for item in data_dirs:
            for part in str(item).split(","):
                part = part.strip()
                if part:
                    dirs.append(part)
    if data_dir:
        for part in str(data_dir).split(","):
            part = part.strip()
            if part:
                dirs.append(part)
    # de-dupe preserving order
    seen = set()
    out: List[str] = []
    for d in dirs:
        key = os.path.abspath(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    if not out:
        print("[ERR] need --data_dir and/or --data_dirs", file=sys.stderr)
        sys.exit(1)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Finetune VGGT depth_head on COLMAP sparse depth")
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Single scene root (or comma-separated). Use with or instead of --data_dirs",
    )
    p.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=None,
        help="One or more scene roots (space- or comma-separated). Concatenated for training",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Source weights (read-only). Default: 3DGS/vggt/models/model.pt",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output depth_head-only checkpoint path (must NOT be model.pt)",
    )
    p.add_argument(
        "--exclude_sparse",
        type=str,
        default=None,
        help="e.g. sparse_9.txt — only applied when a single scene is used",
    )
    p.add_argument("--use_mask", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_side", type=int, default=518, help="Downscale longer side before pad")
    p.add_argument("--min_valid", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    resume = args.resume
    if resume is None:
        resume = os.path.join(_VGGT_DIR, "models", "model.pt")
    out_path = assert_safe_checkpoint_out(args.out, resume)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    scene_dirs = resolve_data_dirs(args.data_dir, args.data_dirs)
    frames: List[FrameSparseDepth] = []
    if len(scene_dirs) == 1:
        exclude = load_exclude_indices(args.exclude_sparse)
        frames = iter_scene_sparse_depths(
            scene_dirs[0],
            exclude_indices=exclude,
            use_mask=args.use_mask,
        )
        print(f"[data] scene={scene_dirs[0]} exclude_sparse={args.exclude_sparse}")
    else:
        if args.exclude_sparse:
            print(
                "[warn] --exclude_sparse ignored for multi-scene training "
                f"(got {args.exclude_sparse})",
                file=sys.stderr,
            )
        for d in scene_dirs:
            sparse0 = os.path.join(d, "sparse", "0")
            if not os.path.isdir(sparse0):
                print(f"[ERR] missing COLMAP sparse/0 under {d}", file=sys.stderr)
                sys.exit(1)
            scene_frames = iter_scene_sparse_depths(
                d,
                exclude_indices=None,
                use_mask=args.use_mask,
            )
            print(f"[data] scene={d} frames={len(scene_frames)}")
            frames.extend(scene_frames)

    dataset = ColmapSparseDepthDataset(
        frames,
        max_side=args.max_side,
        min_valid=args.min_valid,
    )
    print(
        f"[data] scenes={len(scene_dirs)} total={len(frames)} "
        f"excluded={sum(f.excluded for f in frames)} "
        f"train={len(dataset)} min_valid={args.min_valid}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print(f"[model] load resume (read-only): {resume}")
    model = VGGT()
    state = torch.load(resume, map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith("aggregator") or k.startswith("depth_head") for k in state
    ):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (showing up to 5) {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing up to 5) {unexpected[:5]}")
    model.to(device)
    freeze_all_but_depth_head(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable params={n_train:,} / {n_all:,} (depth_head only)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        avg = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            scaler,
            use_amp=use_amp,
            min_valid=max(50, args.min_valid // 4),
            grad_clip=args.grad_clip,
        )
        print(f"[epoch {epoch}/{args.epochs}] mean_loss={avg:.6f}")
        if args.save_every_epoch:
            ep_path = out_path.with_name(out_path.stem + f"_ep{epoch}" + out_path.suffix)
            assert_safe_checkpoint_out(str(ep_path), resume)
            save_depth_head_only(model, ep_path, base_model=resume)
            sz = ep_path.stat().st_size / (1024**2)
            print(f"[save] depth_head-only {ep_path} ({sz:.1f} MiB)")

    save_depth_head_only(model, out_path, base_model=resume)
    sz = out_path.stat().st_size / (1024**2)
    print(f"[done] wrote depth_head-only {out_path} ({sz:.1f} MiB)")
    print(f"[done] original resume untouched: {Path(resume).resolve()}")
    print("[done] downstream: load base model.pt then overlay this file")


if __name__ == "__main__":
    main()
