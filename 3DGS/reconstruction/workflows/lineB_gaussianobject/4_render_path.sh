#!/bin/bash
# [4/10] 粗模型路径视频（默认被 0_run_all SKIP_PATH 跳过；要图再单独跑）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
lineb_activate

echo "=========================================="
echo "[4/10] Render path → ${GS_DIR}"
echo "=========================================="

python render.py \
    -m "${GS_DIR}" \
    -s "${DATA}" \
    --sparse_view_num "${SPARSE_VIEW_NUM}" \
    --sh_degree 2 \
    --init_pcd_name "${INIT_PCD}" \
    --white_background \
    --render_path

echo "[4/10] 完成"
