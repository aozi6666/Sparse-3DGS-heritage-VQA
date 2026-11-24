#!/usr/bin/env python
"""Convert local OCRBench parquet → VLMEvalKit OCRBench.tsv (LMUData)."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import pandas as pd
from PIL import Image


def to_b64(img_obj) -> str:
    raw = img_obj["bytes"] if isinstance(img_obj, dict) and "bytes" in img_obj else img_obj
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError(f"unexpected image type: {type(img_obj)}")
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "OCRBench" / "test-00000-of-00001.parquet",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "lmu" / "OCRBench.tsv",
    )
    args = ap.parse_args()
    df = pd.read_parquet(args.parquet)
    rows = []
    for i, r in df.iterrows():
        ans = r["answer"]
        if hasattr(ans, "tolist"):
            ans = ans.tolist()
        elif not isinstance(ans, list):
            ans = [ans]
        rows.append(
            {
                "index": int(i),
                "image": to_b64(r["image"]),
                "question": str(r["question"]),
                "answer": str(ans),
                "category": str(r["question_type"]),
            }
        )
    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out} n={len(out_df)} size_mb={args.out.stat().st_size / 1e6:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
