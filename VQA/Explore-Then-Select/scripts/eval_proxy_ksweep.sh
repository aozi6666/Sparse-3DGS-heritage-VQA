#!/usr/bin/env bash
# Proxy keep_ratio sweep: N=200, modes=prefix+random+queryip.
# Default ratios: 0.3 0.5 0.9  (+ seed 0.75 from n200).
# Resume-safe: skip ratio if summary already complete; rebuild master from summaries.
# Disk: KEEP_JSONL=0 (default) deletes per-ratio *.jsonl after success (summary only).
# Agent does NOT run — you nohup.
#
# Usage:
#   nohup ./VQA/Explore-Then-Select/scripts/eval_proxy_ksweep.sh \
#     > VQA/eval_results/logs/proxy_ksweep_nohup.out 2>&1 &
#   # interrupt / crash → 再执行同一条命令即可断点续跑
#   KEEP_JSONL=1 ...   # 保留逐题 jsonl（更占盘）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

VENV_BIN="${VENV_BIN:-/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin}"
PYTHON="${VENV_BIN}/python"
export PATH="${VENV_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/VQA/mage_vl/models:${REPO_ROOT}/VQA/Explore-Then-Select:${PYTHONPATH:-}"
export MAGE_VIDEOMME_ROOT="${MAGE_VIDEOMME_ROOT:-${REPO_ROOT}/VQA/mage_vl/data/VideoMME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

N="${N:-200}"
SEED="${SEED:-0}"
MODES="${MODES:-prefix,random,queryip}"
RATIOS="${RATIOS:-0.3 0.5 0.9}"
KEEP_JSONL="${KEEP_JSONL:-0}"

OUT_DIR="${REPO_ROOT}/VQA/eval_results/queryip_samek/ksweep_proxy"
LOG_DIR="${REPO_ROOT}/VQA/eval_results/logs"
MASTER_JSONL="${OUT_DIR}/Mage-VL-QueryIP_proxy_ksweep.jsonl"
MASTER_CSV="${OUT_DIR}/Mage-VL-QueryIP_proxy_ksweep.csv"
PY_EVAL="${REPO_ROOT}/VQA/Explore-Then-Select/mage_query_ip/eval_short60_samek.py"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if ! "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "[error] CUDA not available" >&2
  exit 1
fi

ratio_tag() {
  "${PYTHON}" -c "print('kr{:03d}'.format(int(round(float('${1}') * 100))))"
}

summary_path() {
  echo "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_summary_${1}.json"
}

jsonl_path() {
  echo "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_${1}.jsonl"
}

# Return 0 if summary is complete for this N + MODES.
is_done() {
  local sum="$1"
  [[ -f "${sum}" ]] || return 1
  "${PYTHON}" - "${sum}" "${N}" "${MODES}" <<'PY'
import json, sys
from pathlib import Path
p, n_exp, modes_s = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
want = [m.strip() for m in modes_s.split(",") if m.strip()]
row = json.loads(p.read_text(encoding="utf-8"))
if int(row.get("n") or 0) != n_exp:
    raise SystemExit(1)
acc = row.get("acc") or {}
if any(m not in acc for m in want):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

rebuild_master() {
  "${PYTHON}" - "${OUT_DIR}" "${MASTER_JSONL}" "${MASTER_CSV}" "${N}" <<'PY'
import csv, json, sys
from pathlib import Path
out_dir, master, csv_path = map(Path, sys.argv[1:4])
n_exp = int(sys.argv[4])
by = {}
for p in sorted(out_dir.glob("Mage-VL-QueryIP_short60_samek4_summary_*.json")):
    row = json.loads(p.read_text(encoding="utf-8"))
    if int(row.get("n") or 0) != n_exp:
        continue
    by[float(row["keep_ratio"])] = row
master.write_text(
    "".join(json.dumps(by[k], ensure_ascii=False) + "\n" for k in sorted(by)),
    encoding="utf-8",
)
modes = []
for r in by.values():
    for m in r.get("modes") or []:
        if m not in modes:
            modes.append(m)
fields = ["keep_ratio", "tag", "n", "seed"] + [f"acc_{m}" for m in modes] + [f"correct_{m}" for m in modes]
rows = []
for kr in sorted(by):
    r = by[kr]
    acc, cor = r.get("acc") or {}, r.get("correct") or {}
    out = {"keep_ratio": kr, "tag": r.get("tag") or "", "n": r.get("n"), "seed": r.get("seed")}
    for m in modes:
        out[f"acc_{m}"] = acc.get(m, "")
        out[f"correct_{m}"] = cor.get(m, "")
    rows.append(out)
with csv_path.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"[proxy_ksweep] rebuild master rows={len(rows)} -> {csv_path}")
PY
}

maybe_drop_jsonl() {
  local tag="$1"
  if [[ "${KEEP_JSONL}" == "1" ]]; then
    return 0
  fi
  local jl
  jl="$(jsonl_path "${tag}")"
  if [[ -f "${jl}" ]]; then
    rm -f "${jl}"
    echo "[proxy_ksweep] dropped jsonl (KEEP_JSONL=0): ${jl}"
  fi
}

# Seed 0.75 from n200 hybrid once (tiny summary only).
SEED_TAG="n${N}_kr075"
SEED_SUM_DST="$(summary_path "${SEED_TAG}")"
SEED_SRC="${REPO_ROOT}/VQA/eval_results/queryip_samek/n200/Mage-VL-QueryIP_short60_samek4_summary_n200_hybrid.json"
if [[ ! -f "${SEED_SUM_DST}" && -f "${SEED_SRC}" ]]; then
  "${PYTHON}" - "${SEED_SRC}" "${SEED_SUM_DST}" "${SEED_TAG}" <<'PY'
import json, sys
from pathlib import Path
src, dst, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
row = json.loads(src.read_text(encoding="utf-8"))
want = [m for m in ("prefix", "random", "queryip") if m in (row.get("modes") or [])]
if want:
    row["modes"] = want
    row["acc"] = {m: row["acc"][m] for m in want if m in row.get("acc", {})}
    row["correct"] = {m: row["correct"][m] for m in want if m in row.get("correct", {})}
row["tag"] = f"{tag}_seed"
row["source"] = "n200_hybrid"
row.pop("jsonl", None)
dst.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[proxy_ksweep] seeded 0.75 -> {dst}")
PY
elif [[ -f "${SEED_SUM_DST}" ]]; then
  echo "[proxy_ksweep] 0.75 seed summary already present — skip"
else
  echo "[proxy_ksweep] warn: no n200 hybrid to seed 0.75" >&2
fi

echo "[proxy_ksweep] N=${N} MODES=${MODES} RATIOS=${RATIOS} KEEP_JSONL=${KEEP_JSONL}"
echo "[proxy_ksweep] out=${OUT_DIR} (resume: re-run same command)"

for kr in ${RATIOS}; do
  tag="n${N}_$(ratio_tag "${kr}")"
  sum="$(summary_path "${tag}")"
  echo ""
  if is_done "${sum}"; then
    echo "===== SKIP KEEP_RATIO=${kr} tag=${tag} (summary ok) ====="
    maybe_drop_jsonl "${tag}"
    continue
  fi
  # Incomplete leftover from a crashed run: drop partial jsonl so we start clean.
  rm -f "$(jsonl_path "${tag}")"
  echo "===== RUN KEEP_RATIO=${kr} tag=${tag} $(date -Iseconds) ====="
  "${PYTHON}" "${PY_EVAL}" \
    --n "${N}" \
    --keep-ratio "${kr}" \
    --seed "${SEED}" \
    --modes "${MODES}" \
    --outdir "${OUT_DIR}" \
    --tag "${tag}"
  if ! is_done "${sum}"; then
    echo "[error] expected complete summary missing: ${sum}" >&2
    exit 1
  fi
  maybe_drop_jsonl "${tag}"
done

rebuild_master

echo ""
echo "===== proxy_ksweep ALL DONE $(date -Iseconds) ====="
echo "[proxy_ksweep] ${MASTER_CSV}"
cat "${MASTER_CSV}"
