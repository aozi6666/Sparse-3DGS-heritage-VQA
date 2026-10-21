#!/usr/bin/env python3
"""Report per-image COLMAP sparse-depth pixel counts (no training)."""

from __future__ import annotations

import argparse
import csv
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from colmap_sparse_depth import (  # noqa: E402
    iter_scene_sparse_depths,
    load_exclude_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export COLMAP sparse-depth validity stats for a scene."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Scene root, e.g. .../data/mip360/kitchen",
    )
    parser.add_argument(
        "--exclude_sparse",
        type=str,
        default=None,
        help="Optional sparse_*.txt whose indices are marked excluded (e.g. sparse_9.txt)",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        help="Intersect sparse depth with masks/",
    )
    parser.add_argument(
        "--min_valid",
        type=int,
        default=200,
        help="Frames with fewer valid pixels than this are flagged as skip candidates",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default=None,
        help="Optional CSV path for per-frame stats",
    )
    args = parser.parse_args()

    exclude = load_exclude_indices(args.exclude_sparse)
    frames = iter_scene_sparse_depths(
        args.data_dir,
        exclude_indices=exclude,
        use_mask=args.use_mask,
    )

    train_frames = [f for f in frames if not f.excluded]
    excluded_frames = [f for f in frames if f.excluded]
    skip_candidates = [f for f in train_frames if f.n_valid < args.min_valid]

    print(f"data_dir={args.data_dir}")
    print(f"total_images={len(frames)}")
    print(f"excluded={len(excluded_frames)} (from {args.exclude_sparse})")
    print(f"train_candidates={len(train_frames)}")
    if train_frames:
        n_vals = [f.n_valid for f in train_frames]
        print(
            "train_valid_pixels: "
            f"min={min(n_vals)} median={int(sorted(n_vals)[len(n_vals)//2])} "
            f"max={max(n_vals)} mean={sum(n_vals)/len(n_vals):.1f}"
        )
    print(f"skip_candidates_n_valid<{args.min_valid}: {len(skip_candidates)}")
    if skip_candidates:
        for f in skip_candidates[:20]:
            print(f"  SKIP? idx={f.index} name={f.name} n_valid={f.n_valid}")
        if len(skip_candidates) > 20:
            print(f"  ... and {len(skip_candidates) - 20} more")

    print("--- excluded frames (not for finetune) ---")
    for f in excluded_frames:
        print(f"  idx={f.index} name={f.name} n_valid={f.n_valid}")

    if args.csv_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "index",
                    "image_id",
                    "name",
                    "n_valid",
                    "excluded",
                    "width",
                    "height",
                    "image_path",
                ],
            )
            writer.writeheader()
            for f in frames:
                writer.writerow(
                    {
                        "index": f.index,
                        "image_id": f.image_id,
                        "name": f.name,
                        "n_valid": f.n_valid,
                        "excluded": int(f.excluded),
                        "width": f.width,
                        "height": f.height,
                        "image_path": f.image_path,
                    }
                )
        print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    main()
