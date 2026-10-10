#!/bin/bash
# [3/10] 粗模型测试集评测（只写 results.json）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
MODEL_DIR="${MODEL_DIR:-${GS_DIR}}"
lineb_activate

echo "=========================================="
echo "[3/10] Render test → ${MODEL_DIR}"
echo "=========================================="

python render.py \
    -m "${MODEL_DIR}" \
    -s "${DATA}" \
    --sparse_view_num "${SPARSE_VIEW_NUM}" --sh_degree 2 \
    --init_pcd_name "${INIT_PCD}" \
    --white_background \
    --skip_train \
    --not_generate_video \
    --not_saveimages

echo "[3/10] 完成 → ${MODEL_DIR}/test/*/results.json"
