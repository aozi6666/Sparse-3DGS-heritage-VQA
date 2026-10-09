#!/bin/bash
# [5/10] Leave-One-Out stage1（须保留 chkpnt6000，stage2 依赖）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
lineb_activate

echo "=========================================="
echo "[5/10] Leave-One-Out stage1 → ${LOO_DIR}"
echo "=========================================="

python leave_one_out_stage1.py \
    -s "${DATA}" \
    -m "${LOO_DIR}" \
    -r 4 --sparse_view_num "${SPARSE_VIEW_NUM}" --sh_degree 2 \
    --init_pcd_name "${INIT_PCD}" \
    --white_background --random_background \
    --iterations "${ITERATIONS}" \
    --save_iterations "${ITERATIONS}" \
    --test_iterations "${ITERATIONS}" \
    --checkpoint_iterations 6000

# 省盘：TB / input 副本；保留 chkpnt6000.pth 与最终 ply
find "${LOO_DIR}" -name 'events.out.tfevents.*' -delete 2>/dev/null || true
find "${LOO_DIR}" -name 'input.ply' -delete 2>/dev/null || true

echo "[5/10] 完成 → ${LOO_DIR}/"
