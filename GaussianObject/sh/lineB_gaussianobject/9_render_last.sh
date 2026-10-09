#!/bin/bash
# [9/10] 修复后测试集评测（只写 results.json）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
lineb_activate

[[ -f "${FINAL_PLY}" ]] || { echo "[ERR] missing ${FINAL_PLY}"; exit 1; }

echo "=========================================="
echo "[9/10] Render test (repaired)  ply=${FINAL_PLY}"
echo "=========================================="

python render.py \
    -m "${GS_DIR}" \
    --sparse_view_num "${SPARSE_VIEW_NUM}" --sh_degree 2 \
    --init_pcd_name "${INIT_PCD}" \
    --white_background \
    --load_ply "${FINAL_PLY}" \
    --skip_train \
    --not_generate_video \
    --not_saveimages

echo "[9/10] 完成 → ${GS_DIR}/test/*/results.json"
