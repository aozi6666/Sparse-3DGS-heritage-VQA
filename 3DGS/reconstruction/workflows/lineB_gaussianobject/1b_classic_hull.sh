#!/bin/bash
# Classic visual hull（无 VGGT）→ visual_hull_${SPARSE_ID}.ply
# 用法：bash 1b_classic_hull.sh
# 已存在则跳过；强制：FORCE_HULL=1 bash 1b_classic_hull.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
lineb_activate

CLASSIC_INIT="visual_hull_${SPARSE_ID}"
PLY="${DATA}/${CLASSIC_INIT}.ply"
FORCE_HULL="${FORCE_HULL:-0}"

echo "=========================================="
echo "[1b] Classic visual hull → ${PLY}"
echo "=========================================="

if [[ -f "${PLY}" && "${FORCE_HULL}" != "1" ]]; then
  echo "[1b] skip (exists). FORCE_HULL=1 to regenerate."
  exit 0
fi

# 禁止误伤 VGGT 基线 ply
VGGT_PLY="${DATA}/visual_hull_vggt_enhanced_${SPARSE_ID}.ply"
[[ -f "${VGGT_PLY}" ]] || echo "[WARN] missing VGGT baseline ply: ${VGGT_PLY}"

python visual_hull.py \
  --data_dir "${DATA}" \
  --sparse_id "${SPARSE_ID}" \
  --reso 2 \
  --not_vis

[[ -f "${PLY}" ]] || { echo "[ERR] missing ${PLY} after visual_hull.py"; exit 1; }
echo "[1b] 完成 → ${PLY}"
