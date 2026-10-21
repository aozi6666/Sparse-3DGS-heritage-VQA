#!/usr/bin/env python3
"""Offline asserts for Cultural LoRA disk / checkpoint policy (3 stages)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths import (
    CIQI_MANIFEST,
    CIQI_SUBSET_JSONL,
    DEFAULT_CIQI_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_MSVD_DIR,
    DEFAULT_OUTPUT_DIR,
    MSVD_MANIFEST,
    MSVD_SUBSET_DIR,
)


class VerifyError(Exception):
    pass


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str) -> None:
    raise VerifyError(msg)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def verify_subset(ciqi_dir: Path, msvd_dir: Path, expect_ciqi: int, expect_msvd: int) -> None:
    ciqi_jsonl = ciqi_dir / CIQI_SUBSET_JSONL
    ciqi_man = ciqi_dir / CIQI_MANIFEST
    if not ciqi_jsonl.is_file():
        _fail(f"missing {ciqi_jsonl}")
    if not ciqi_man.is_file():
        _fail(f"missing {ciqi_man}")
    n = _count_jsonl(ciqi_jsonl)
    man = json.loads(ciqi_man.read_text(encoding="utf-8"))
    if n != expect_ciqi:
        _fail(f"CiQi subset lines={n}, expect {expect_ciqi}")
    if man.get("n") != expect_ciqi:
        _fail(f"CiQi manifest n={man.get('n')}, expect {expect_ciqi}")
    missing = [rel for rel in man["image_refs"] if not (ciqi_dir / rel).is_file()]
    if missing:
        _fail(f"CiQi missing {len(missing)} images e.g. {missing[:3]}")
    _ok(f"CiQi subset {n} rows, {len(man['image_refs'])} images present")

    msvd_man_path = msvd_dir / MSVD_MANIFEST
    qa = msvd_dir / MSVD_SUBSET_DIR / "qa.jsonl"
    frames = msvd_dir / MSVD_SUBSET_DIR / "frames"
    if not msvd_man_path.is_file():
        _fail(f"missing {msvd_man_path}")
    if not qa.is_file():
        _fail(f"missing {qa}")
    mman = json.loads(msvd_man_path.read_text(encoding="utf-8"))
    nq = _count_jsonl(qa)
    if nq != expect_msvd:
        _fail(f"MSVD qa lines={nq}, expect {expect_msvd}")
    if mman.get("n") != expect_msvd:
        _fail(f"MSVD manifest n={mman.get('n')}, expect {expect_msvd}")
    if not frames.is_dir():
        _fail(f"missing {frames}")
    # spot-check first few frame files
    checked = 0
    with qa.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            fp = msvd_dir / MSVD_SUBSET_DIR / rec["frames_file"]
            if not fp.is_file():
                _fail(f"missing frames {fp}")
            checked += 1
            if checked >= 20:
                break
    _ok(f"MSVD subset {nq} QA, frames dir ok (spot-checked {checked})")


def verify_after_delete(ciqi_dir: Path, msvd_dir: Path, expect_ciqi: int, expect_msvd: int) -> None:
    verify_subset(ciqi_dir, msvd_dir, expect_ciqi, expect_msvd)

    man = json.loads((ciqi_dir / CIQI_MANIFEST).read_text(encoding="utf-8"))
    keep = set(man["image_refs"])
    images_root = ciqi_dir / "images"
    leftovers = []
    if images_root.is_dir():
        for p in images_root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(ciqi_dir).as_posix()
            if rel not in keep:
                leftovers.append(rel)
    if leftovers:
        _fail(f"CiQi still has {len(leftovers)} non-subset images e.g. {leftovers[:3]}")
    _ok("CiQi hard-delete: no unused images left")

    train_jsonl = ciqi_dir / "train.jsonl"
    if train_jsonl.is_file():
        tn = _count_jsonl(train_jsonl)
        if tn != expect_ciqi:
            _fail(f"after delete, train.jsonl has {tn} rows, expect {expect_ciqi}")
        _ok(f"CiQi train.jsonl replaced ({tn} rows)")

    shards = sorted(msvd_dir.glob("train-*-of-*.parquet"))
    if shards:
        _fail(f"MSVD original shards still present: {[p.name for p in shards]}")
    _ok("MSVD original shards deleted")

    if not (msvd_dir / MSVD_SUBSET_DIR / "qa.jsonl").is_file():
        _fail("MSVD train_8k/qa.jsonl missing after delete")
    _ok("MSVD train_8k retained")


def verify_after_smoke(output_dir: Path, model_dir: Path) -> None:
    output_dir = output_dir.resolve()
    model_dir = model_dir.resolve()
    if not output_dir.is_dir():
        _fail(f"missing output_dir {output_dir}")

    adapter = output_dir / "adapter"
    if not adapter.is_dir():
        _fail(f"missing adapter dir {adapter}")

    # PEFT marker files
    has_adapter_cfg = (adapter / "adapter_config.json").is_file()
    has_weights = (
        (adapter / "adapter_model.safetensors").is_file()
        or (adapter / "adapter_model.bin").is_file()
        or any(adapter.glob("adapter_model*.safetensors"))
    )
    if not has_adapter_cfg:
        _fail("adapter_config.json missing")
    if not has_weights:
        _fail("adapter weight file missing")
    _ok("PEFT adapter files present")

    # No full base shards in output tree
    leaked = list(output_dir.rglob("model-*-of-*.safetensors")) + list(
        output_dir.rglob("pytorch_model*.bin")
    )
    if leaked:
        _fail(f"full model weights found under output: {leaked[:5]}")
    _ok("no full base weight shards under output")

    # At most one checkpoint-* (prefer zero after final prune) + adapter
    ckpts = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if len(ckpts) > 1:
        _fail(f"more than 1 checkpoint-* dirs: {[p.name for p in ckpts]}")
    _ok(f"checkpoint-* count={len(ckpts)} (allowed <=1)")

    # Output must not be a copy of the base model dir
    if output_dir == model_dir:
        _fail("output_dir equals base model dir")
    base_index = model_dir / "model.safetensors.index.json"
    if not base_index.is_file():
        _fail(f"base model index missing at {base_index} (did something delete local weights?)")
    _ok(f"base weights untouched at {model_dir}")

    # Adapter should be much smaller than one base shard
    adapter_bytes = sum(p.stat().st_size for p in adapter.rglob("*") if p.is_file())
    base_bytes = sum(
        p.stat().st_size for p in model_dir.glob("model-*.safetensors") if p.is_file()
    )
    if base_bytes > 0 and adapter_bytes > base_bytes * 0.25:
        _fail(
            f"adapter size {adapter_bytes/1e9:.2f}G suspiciously large vs base shards {base_bytes/1e9:.2f}G"
        )
    _ok(f"adapter size {adapter_bytes/1e6:.1f} MB (base shards {base_bytes/1e9:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        required=True,
        choices=["subset", "after_delete", "after_smoke"],
    )
    ap.add_argument("--ciqi-dir", type=Path, default=DEFAULT_CIQI_DIR)
    ap.add_argument("--msvd-dir", type=Path, default=DEFAULT_MSVD_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "smoke")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--expect-ciqi", type=int, default=16000)
    ap.add_argument("--expect-msvd", type=int, default=8000)
    args = ap.parse_args()

    try:
        if args.stage == "subset":
            verify_subset(args.ciqi_dir, args.msvd_dir, args.expect_ciqi, args.expect_msvd)
        elif args.stage == "after_delete":
            verify_after_delete(
                args.ciqi_dir, args.msvd_dir, args.expect_ciqi, args.expect_msvd
            )
        else:
            verify_after_smoke(args.output_dir, args.model)
    except VerifyError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[DONE] stage={args.stage} passed")


if __name__ == "__main__":
    main()
