#!/bin/bash
# [10/10] 修复后路径视频（默认 SKIP_PATH 跳过）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
lineb_activate

[[ -f "${FINAL_PLY}" ]] || { echo "[ERR] missing ${FINAL_PLY}"; exit 1; }

echo "=========================================="
echo "[10/10] Render path (repaired)"
echo "=========================================="

python render.py \
    -m "${GS_DIR}" \
    -s "${DATA}" \
    --sparse_view_num "${SPARSE_VIEW_NUM}" --sh_degree 2 \
    --init_pcd_name "${INIT_PCD}" \
    --white_background --render_path \
    --load_ply "${FINAL_PLY}"

echo "[10/10] 完成"
