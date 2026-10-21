#!/usr/bin/env bash
# Batch keep_ratio sweep on Video-MME short-60 (4 modes).
# Default ratios: 0.5 0.6 0.9 (0.75 already done — skipped unless INCLUDE_075=1).
# Agent does NOT run this — you execute once and leave.
#
# Usage:
#   bash VQA/Explore-Then-Select/scripts/eval_short60_ksweep.sh
#   RATIOS="0.5 0.6 0.9" INCLUDE_075=1 bash .../eval_short60_ksweep.sh
#
# Per-ratio backups + one master table for later plotting:
#   VQA/eval_results/Mage-VL-QueryIP_short60_samek4_krXXX.jsonl
#   VQA/eval_results/Mage-VL-QueryIP_short60_samek4_summary_krXXX.json
#   VQA/eval_results/Mage-VL-QueryIP_short60_ksweep.jsonl   (append one line / ratio)
#   VQA/eval_results/Mage-VL-QueryIP_short60_ksweep.csv      (rebuild each time)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${REPO_ROOT}/VQA/eval_results"
EVAL_SH="${REPO_ROOT}/VQA/Explore-Then-Select/scripts/eval_short60_samek.sh"
MASTER_JSONL="${OUT_DIR}/Mage-VL-QueryIP_short60_ksweep.jsonl"
MASTER_CSV="${OUT_DIR}/Mage-VL-QueryIP_short60_ksweep.csv"
LOG="${OUT_DIR}/Mage-VL-QueryIP_short60_ksweep_run.log"

VENV_BIN="${VENV_BIN:-/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin}"
PYTHON="${VENV_BIN}/python"

mkdir -p "${OUT_DIR}"
touch "${MASTER_JSONL}"

ratio_tag() {
  # 0.5 -> kr050 ; 0.75 -> kr075 ; 0.9 -> kr090
  "${PYTHON}" -c "print('kr{:03d}'.format(int(round(float('${1}') * 100))))"
}

seed_075_into_master() {
  local sum="${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_summary.json"
  local tag="kr075"
  if [[ ! -f "${sum}" ]]; then
    echo "[ksweep] no existing 0.75 summary; skip seed" >&2
    return 0
  fi
  # Copy backups if missing
  if [[ ! -f "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_${tag}.jsonl" ]] \
     && [[ -f "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4.jsonl" ]]; then
    cp -n "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4.jsonl" \
      "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_${tag}.jsonl" || true
  fi
  if [[ ! -f "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_summary_${tag}.json" ]]; then
    cp -n "${sum}" "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_summary_${tag}.json" || true
  fi
  # Append 0.75 row if not already in master
  if grep -q '"keep_ratio": 0.75' "${MASTER_JSONL}" 2>/dev/null \
     || grep -q '"keep_ratio":0.75' "${MASTER_JSONL}" 2>/dev/null; then
    echo "[ksweep] master already has keep_ratio=0.75"
  else
    "${PYTHON}" - "${sum}" "${MASTER_JSONL}" <<'PY'
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
row = json.loads(src.read_text(encoding="utf-8"))
row["tag"] = "kr075"
row["source"] = "seed_from_existing_075"
with dst.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("[ksweep] seeded keep_ratio=0.75 into master")
PY
  fi
}

append_summary() {
  local tag="$1"
  local sum="${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_summary.json"
  "${PYTHON}" - "${sum}" "${MASTER_JSONL}" "${tag}" <<'PY'
import json, sys
from pathlib import Path
src, dst, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
row = json.loads(src.read_text(encoding="utf-8"))
row["tag"] = tag
row["source"] = "ksweep"
# drop duplicate keep_ratio lines then rewrite? append only; rebuild csv later
with dst.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[ksweep] appended {tag} keep_ratio={row.get('keep_ratio')} acc={row.get('acc')}")
PY
}

rebuild_csv() {
  "${PYTHON}" - "${MASTER_JSONL}" "${MASTER_CSV}" <<'PY'
import json, csv, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
# last row wins per keep_ratio
by_kr = {}
if src.is_file():
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        by_kr[float(row["keep_ratio"])] = row
modes = ["prefix", "random", "queryip", "queryip_bitcost"]
fields = ["keep_ratio", "tag", "n", "seed"] + [f"acc_{m}" for m in modes] + [f"correct_{m}" for m in modes]
rows = []
for kr in sorted(by_kr):
    r = by_kr[kr]
    acc = r.get("acc") or {}
    cor = r.get("correct") or {}
    out = {
        "keep_ratio": kr,
        "tag": r.get("tag", ""),
        "n": r.get("n", ""),
        "seed": r.get("seed", ""),
    }
    for m in modes:
        out[f"acc_{m}"] = acc.get(m, "")
        out[f"correct_{m}"] = cor.get(m, "")
    rows.append(out)
with dst.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"[ksweep] wrote {dst} ({len(rows)} ratios)")
PY
}

# --- main ---
RATIOS_DEFAULT="0.5 0.6 0.9"
if [[ "${INCLUDE_075:-0}" == "1" ]]; then
  RATIOS="${RATIOS:-0.5 0.6 0.75 0.9}"
else
  RATIOS="${RATIOS:-${RATIOS_DEFAULT}}"
fi

echo "[ksweep] repo=${REPO_ROOT}"
echo "[ksweep] ratios=${RATIOS}"
echo "[ksweep] log=${LOG}"
echo "[ksweep] master=${MASTER_JSONL}"
{
  echo "===== ksweep start $(date -Iseconds) ratios=${RATIOS} ====="
} | tee -a "${LOG}"

seed_075_into_master
rebuild_csv

chmod +x "${EVAL_SH}"

for kr in ${RATIOS}; do
  tag="$(ratio_tag "${kr}")"
  echo "" | tee -a "${LOG}"
  echo "===== KEEP_RATIO=${kr} (${tag}) $(date -Iseconds) =====" | tee -a "${LOG}"
  KEEP_RATIO="${kr}" bash "${EVAL_SH}" 2>&1 | tee -a "${LOG}"

  cp -f "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4.jsonl" \
    "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_${tag}.jsonl"
  cp -f "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_summary.json" \
    "${OUT_DIR}/Mage-VL-QueryIP_short60_samek4_summary_${tag}.json"
  append_summary "${tag}"
  rebuild_csv
  echo "[ksweep] done ${tag}" | tee -a "${LOG}"
done

echo "" | tee -a "${LOG}"
echo "===== ksweep ALL DONE $(date -Iseconds) =====" | tee -a "${LOG}"
echo "[ksweep] table: ${MASTER_CSV}"
echo "[ksweep] jsonl: ${MASTER_JSONL}"
cat "${MASTER_CSV}"
