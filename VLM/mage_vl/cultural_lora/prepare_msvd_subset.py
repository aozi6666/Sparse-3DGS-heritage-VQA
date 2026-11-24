#!/usr/bin/env python3
"""Sample MSVD-QA to N QA pairs; write compact train_8k/; optional hard-delete shards."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paths import DEFAULT_MSVD_DIR, MSVD_MANIFEST, MSVD_SUBSET_DIR

SHARD_RE = re.compile(r"train-\d{5}-of-\d{5}\.parquet$")


def list_shards(msvd_dir: Path) -> list[Path]:
    return sorted(p for p in msvd_dir.glob("train-*.parquet") if SHARD_RE.search(p.name))


def shard_readable(con: duckdb.DuckDBPyConnection, path: Path) -> bool:
    """Probe real data pages (count(*) alone can miss truncated shards)."""
    try:
        con.execute(
            f"""
            SELECT video_path, qa, num_frames, height, width, channels, binary_frames
            FROM read_parquet('{path.as_posix()}')
            LIMIT 1
            """
        ).fetchone()
        return True
    except Exception as e:
        print(f"[msvd] skip unreadable shard {path.name}: {e}")
        return False


def expand_qa_index(con: duckdb.DuckDBPyConnection, shards: list[Path]) -> list[dict]:
    """Return list of {shard, row_idx, qa_idx, question, answer, video_path}."""
    entries: list[dict] = []
    for shard in shards:
        # DuckDB rows are 0-based via ROW_NUMBER
        q = f"""
        SELECT
          row_number() OVER () - 1 AS row_idx,
          video_path,
          qa
        FROM read_parquet('{shard.as_posix()}')
        """
        rows = con.execute(q).fetchall()
        for row_idx, video_path, qa in rows:
            if not qa:
                continue
            for qa_idx, pair in enumerate(qa):
                if not pair or len(pair) < 2:
                    continue
                entries.append(
                    {
                        "shard": shard.name,
                        "row_idx": int(row_idx),
                        "qa_idx": int(qa_idx),
                        "video_path": video_path,
                        "question": str(pair[0]),
                        "answer": str(pair[1]),
                    }
                )
    return entries


def video_key(video_path: str, shard: str, row_idx: int) -> str:
    stem = Path(str(video_path)).stem or f"row{row_idx}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:80]
    return f"{Path(shard).stem}__{row_idx:05d}__{safe}"


def load_video_blob(
    con: duckdb.DuckDBPyConnection, shard: Path, row_idx: int
) -> tuple[bytes, int, int, int, int]:
    # ORDER BY row_number approach: fetch with LIMIT/OFFSET
    q = f"""
    SELECT binary_frames, num_frames, height, width, channels
    FROM read_parquet('{shard.as_posix()}')
    LIMIT 1 OFFSET {int(row_idx)}
    """
    bf, nf, h, w, c = con.execute(q).fetchone()
    return bf, int(nf), int(h), int(w), int(c)


def subsample_frames(
    bf: bytes, nf: int, h: int, w: int, c: int, max_frames: int
) -> np.ndarray:
    arr = np.frombuffer(bf, dtype=np.uint8)
    expect = nf * h * w * c
    if arr.size != expect:
        raise ValueError(f"binary_frames size {arr.size} != {expect}")
    frames = arr.reshape(nf, h, w, c)
    if nf <= max_frames:
        return frames.copy()
    idx = np.linspace(0, nf - 1, max_frames, dtype=int)
    return frames[idx].copy()


def write_subset(
    msvd_dir: Path,
    entries: list[dict],
    seed: int,
    n: int,
    max_frames: int,
) -> Path:
    out_dir = msvd_dir / MSVD_SUBSET_DIR
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    qa_path = out_dir / "qa.jsonl"

    # Unique videos needed
    needed_videos: dict[tuple[str, int], list[dict]] = {}
    for e in entries:
        key = (e["shard"], e["row_idx"])
        needed_videos.setdefault(key, []).append(e)

    con = duckdb.connect()
    shard_map = {p.name: p for p in list_shards(msvd_dir)}
    video_meta: dict[str, dict] = {}

    for (shard_name, row_idx), group in tqdm(
        needed_videos.items(), desc="export frames"
    ):
        shard = shard_map[shard_name]
        vk = video_key(group[0]["video_path"], shard_name, row_idx)
        npy_rel = f"frames/{vk}.npy"
        npy_abs = out_dir / npy_rel
        if not npy_abs.is_file():
            bf, nf, h, w, c = load_video_blob(con, shard, row_idx)
            frames = subsample_frames(bf, nf, h, w, c, max_frames=max_frames)
            np.save(npy_abs, frames)
            video_meta[vk] = {
                "video_key": vk,
                "source_shard": shard_name,
                "source_row": row_idx,
                "video_path": group[0]["video_path"],
                "frames_file": npy_rel,
                "num_frames": int(frames.shape[0]),
                "height": int(frames.shape[1]),
                "width": int(frames.shape[2]),
                "channels": int(frames.shape[3]),
                "orig_num_frames": nf,
            }
        else:
            # already exported in this run
            if vk not in video_meta:
                arr = np.load(npy_abs, mmap_mode="r")
                video_meta[vk] = {
                    "video_key": vk,
                    "source_shard": shard_name,
                    "source_row": row_idx,
                    "video_path": group[0]["video_path"],
                    "frames_file": npy_rel,
                    "num_frames": int(arr.shape[0]),
                    "height": int(arr.shape[1]),
                    "width": int(arr.shape[2]),
                    "channels": int(arr.shape[3]),
                }
        for e in group:
            e["video_key"] = vk
            e["frames_file"] = npy_rel

    with qa_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(
                json.dumps(
                    {
                        "video_key": e["video_key"],
                        "frames_file": e["frames_file"],
                        "question": e["question"],
                        "answer": e["answer"],
                        "source_shard": e["shard"],
                        "source_row": e["row_idx"],
                        "source_qa_idx": e["qa_idx"],
                        "video_path": e["video_path"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    videos_json = out_dir / "videos.json"
    videos_json.write_text(
        json.dumps(list(video_meta.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_shard_names = [p.name for p in list_shards(msvd_dir)]
    used_shards = sorted({e["shard"] for e in entries})
    manifest = {
        "dataset": "MSVD-QA",
        "seed": seed,
        "n": n,
        "num_qa": len(entries),
        "num_videos": len(video_meta),
        "max_frames_stored": max_frames,
        "subset_dir": MSVD_SUBSET_DIR,
        "qa_jsonl": f"{MSVD_SUBSET_DIR}/qa.jsonl",
        "videos_json": f"{MSVD_SUBSET_DIR}/videos.json",
        "used_shards": used_shards,
        "all_shards_at_prep": all_shard_names,
        "note": (
            "QA pairs expanded from nested qa field; frames subsampled for disk. "
            "Unreadable shards (e.g. truncated parquet) are skipped at prep time."
        ),
    }
    man_path = msvd_dir / MSVD_MANIFEST
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[msvd] wrote {qa_path} ({len(entries)} QA)")
    print(f"[msvd] wrote {frames_dir} ({len(video_meta)} videos)")
    print(f"[msvd] wrote {man_path}")
    return out_dir


def hard_delete_shards(msvd_dir: Path) -> None:
    subset = msvd_dir / MSVD_SUBSET_DIR
    man = msvd_dir / MSVD_MANIFEST
    if not subset.is_dir() or not man.is_file():
        raise SystemExit("subset missing; build without --confirm-delete first")
    qa = subset / "qa.jsonl"
    n_qa = sum(1 for _ in qa.open(encoding="utf-8"))
    manifest = json.loads(man.read_text(encoding="utf-8"))
    if n_qa != manifest.get("n"):
        raise SystemExit(f"qa.jsonl lines={n_qa} != manifest n={manifest.get('n')}")

    shards = list_shards(msvd_dir)
    bytes_del = sum(p.stat().st_size for p in shards)
    print(
        f"[msvd] will delete {len(shards)} original shards (~{bytes_del / 1e9:.2f} GB)"
    )
    for p in shards:
        print(f"  rm {p.name}")
        p.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--msvd-dir", type=Path, default=DEFAULT_MSVD_DIR)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Uniform subsample frames per video when exporting (train uses <= this)",
    )
    ap.add_argument("--confirm-delete", action="store_true")
    args = ap.parse_args()
    msvd_dir = args.msvd_dir.resolve()

    if args.confirm_delete:
        hard_delete_shards(msvd_dir)
        print("[msvd] original shards deleted. Keep train_8k/ + subset_manifest_8k.json")
        return

    shards_all = list_shards(msvd_dir)
    if not shards_all:
        raise SystemExit(f"no train-*-of-*.parquet under {msvd_dir}")

    con = duckdb.connect()
    shards = [p for p in shards_all if shard_readable(con, p)]
    if not shards:
        raise SystemExit("no readable MSVD shards (all corrupt?). Re-download dataset.")

    print(f"[msvd] readable shards {len(shards)}/{len(shards_all)}")
    entries = expand_qa_index(con, shards)
    print(f"[msvd] expanded QA pairs: {len(entries)}")
    if len(entries) < args.n:
        raise SystemExit(f"only {len(entries)} QA pairs available, need {args.n}")

    rng = random.Random(args.seed)
    rng.shuffle(entries)
    chosen = entries[: args.n]
    write_subset(msvd_dir, chosen, args.seed, args.n, max_frames=args.max_frames)
    print("[msvd] dry-run complete (no deletes). Re-run with --confirm-delete to remove shards.")


if __name__ == "__main__":
    main()
