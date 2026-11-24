#!/usr/bin/env python3
"""Sample CiQi-VQA to N rows, optionally hard-delete unused images + full jsonl."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths import CIQI_MANIFEST, CIQI_SUBSET_JSONL, DEFAULT_CIQI_DIR


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _collect_image_refs(records: list[dict]) -> set[str]:
    refs: set[str] = set()
    for rec in records:
        for rel in rec.get("images") or []:
            refs.add(str(rel).replace("\\", "/"))
    return refs


def _estimate_bytes(paths: Iterable[Path]) -> int:
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def sample_records(src_jsonl: Path, n: int, seed: int) -> list[dict]:
    records = list(_iter_jsonl(src_jsonl))
    if len(records) < n:
        raise SystemExit(f"CiQi has only {len(records)} rows, need {n}")
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    chosen = [records[i] for i in idx[:n]]
    return chosen


def write_subset(ciqi_dir: Path, records: list[dict], seed: int, n: int) -> Path:
    out_jsonl = ciqi_dir / CIQI_SUBSET_JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    refs = sorted(_collect_image_refs(records))
    missing = []
    for rel in refs:
        if not (ciqi_dir / rel).is_file():
            missing.append(rel)
    if missing:
        raise SystemExit(
            f"{len(missing)} referenced images missing (e.g. {missing[:3]})"
        )

    manifest = {
        "dataset": "CiQi-VQA",
        "seed": seed,
        "n": n,
        "source_jsonl": "train.jsonl",
        "subset_jsonl": CIQI_SUBSET_JSONL,
        "num_records": len(records),
        "num_unique_images": len(refs),
        "image_refs": refs,
        "note": "Fixed-seed random sample; CiQi-Bench exclusion list not applied (not local).",
    }
    man_path = ciqi_dir / CIQI_MANIFEST
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ciqi] wrote {out_jsonl} ({len(records)} rows)")
    print(f"[ciqi] wrote {man_path} ({len(refs)} unique images)")
    return out_jsonl


def hard_delete(ciqi_dir: Path, keep_refs: set[str]) -> None:
    images_root = ciqi_dir / "images"
    if not images_root.is_dir():
        raise SystemExit(f"missing images dir: {images_root}")

    all_files = [p for p in images_root.rglob("*") if p.is_file()]
    keep_paths = {(ciqi_dir / rel).resolve() for rel in keep_refs}
    to_delete = [p for p in all_files if p.resolve() not in keep_paths]
    bytes_del = _estimate_bytes(to_delete)
    print(
        f"[ciqi] will delete {len(to_delete)} unused image files "
        f"(~{bytes_del / 1e9:.2f} GB); keep {len(keep_paths)}"
    )

    for p in to_delete:
        p.unlink(missing_ok=True)

    # Drop empty part dirs
    for d in sorted(images_root.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass

    full_jsonl = ciqi_dir / "train.jsonl"
    subset = ciqi_dir / CIQI_SUBSET_JSONL
    if full_jsonl.is_file() and subset.is_file():
        bak = ciqi_dir / "train.jsonl.full.bak"
        if bak.exists():
            bak.unlink()
        # Replace full with subset content (no second copy of 557k rows)
        shutil.move(str(full_jsonl), str(bak))
        shutil.copy2(subset, full_jsonl)
        bak_size = bak.stat().st_size
        print(f"[ciqi] replaced train.jsonl; full backup at {bak.name} ({bak_size / 1e9:.2f} GB)")
        print("[ciqi] remove backup after verify: "
              f"rm {bak}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ciqi-dir", type=Path, default=DEFAULT_CIQI_DIR)
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--confirm-delete",
        action="store_true",
        help="After subset exists: delete unused images and replace train.jsonl",
    )
    ap.add_argument(
        "--delete-full-backup",
        action="store_true",
        help="Also remove train.jsonl.full.bak after replace (max disk save)",
    )
    args = ap.parse_args()
    ciqi_dir = args.ciqi_dir.resolve()
    src = ciqi_dir / "train.jsonl"
    subset_path = ciqi_dir / CIQI_SUBSET_JSONL
    man_path = ciqi_dir / CIQI_MANIFEST

    if args.confirm_delete:
        if not subset_path.is_file() or not man_path.is_file():
            raise SystemExit("Run without --confirm-delete first to build subset + manifest")
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        if manifest.get("n") != args.n or manifest.get("seed") != args.seed:
            raise SystemExit(
                f"manifest n/seed={manifest.get('n')}/{manifest.get('seed')} "
                f"!= requested {args.n}/{args.seed}; rebuild subset first"
            )
        keep = set(manifest["image_refs"])
        hard_delete(ciqi_dir, keep)
        if args.delete_full_backup:
            bak = ciqi_dir / "train.jsonl.full.bak"
            if bak.is_file():
                print(f"[ciqi] deleting backup {bak} ({bak.stat().st_size / 1e9:.2f} GB)")
                bak.unlink()
        return

    # Prefer sampling from full train.jsonl; if already replaced, sample from it
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    # If already a subset file with fewer lines than n, refuse
    line_count = sum(1 for _ in _iter_jsonl(src))
    if line_count < args.n:
        raise SystemExit(f"{src} has {line_count} rows < n={args.n}")

    records = sample_records(src, args.n, args.seed)
    write_subset(ciqi_dir, records, args.seed, args.n)
    print("[ciqi] dry-run complete (no deletes). Re-run with --confirm-delete to hard-delete.")


if __name__ == "__main__":
    main()
