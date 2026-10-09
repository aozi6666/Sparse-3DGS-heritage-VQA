#!/bin/bash
# [6/10] Leave-One-Out stage2 → diffs.pkl（读 stage1 的 chkpnt6000）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
lineb_activate

echo "=========================================="
echo "[6/10] Leave-One-Out stage2 → ${LOO_DIR}"
echo "=========================================="

python leave_one_out_stage2.py \
    -s "${DATA}" \
    -m "${LOO_DIR}" \
    -r 4 --sparse_view_num "${SPARSE_VIEW_NUM}" --sh_degree 2 \
    --init_pcd_name "${INIT_PCD}" \
    --white_background --random_background \
    --iterations "${ITERATIONS}" \
    --save_iterations "${ITERATIONS}" \
    --test_iterations "${ITERATIONS}"

find "${LOO_DIR}" -name 'events.out.tfevents.*' -delete 2>/dev/null || true

echo "[6/10] 完成"
