#!/usr/bin/env python3
"""Summarize Mage-VL-QueryIP token jsonl (Avg #vis-token, relative to default N)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DEFAULT_JSONL = Path(
    "/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/eval_results/"
    "Mage-VL-QueryIP_videomme_tokens.jsonl"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = ap.parse_args()
    if not args.jsonl.is_file():
        print(f"[error] missing jsonl: {args.jsonl}")
        return 2

    rows = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("[error] empty jsonl")
        return 2

    n_vis = [float(r["n_vis_tokens"]) for r in rows]
    n_def = [float(r.get("n_vis_tokens_default", 0)) for r in rows]
    ratios = [
        (a / b) if b > 0 else float("nan")
        for a, b in zip(n_vis, n_def)
    ]
    ratios_ok = [x for x in ratios if x == x]

    print(f"file={args.jsonl}")
    print(f"n_samples={len(rows)}")
    print(f"avg_n_vis_tokens={statistics.mean(n_vis):.2f}")
    print(f"avg_n_vis_tokens_default={statistics.mean(n_def):.2f}")
    if ratios_ok:
        print(f"avg_token_ratio_vs_default={statistics.mean(ratios_ok):.4f}")
    qtypes = {}
    for r in rows:
        q = r.get("qtype") or "unknown"
        qtypes[q] = qtypes.get(q, 0) + 1
    print(f"qtype_counts={qtypes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
