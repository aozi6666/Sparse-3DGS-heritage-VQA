#!/bin/bash
# [2/10] 粗 3DGS（省盘：默认只存/测最后一轮）
# 可选：SAVE_ITERS / TEST_ITERS 为空格分隔列表，如 "3000 5000 7000 10000"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
lineb_activate

if [[ "${USE_DROP}" == "1" ]]; then
  DROP_FLAG=(--use_drop --drop_rate "${DROP_RATE}" --drop_schedule "${DROP_SCHEDULE}")
else
  DROP_FLAG=()
fi

# Default: only final iteration (compat). Sweep can export SAVE_ITERS / TEST_ITERS.
# shellcheck disable=SC2206
SAVE_ARR=(${SAVE_ITERS:-${ITERATIONS}})
# shellcheck disable=SC2206
TEST_ARR=(${TEST_ITERS:-${ITERATIONS}})

echo "=========================================="
echo "[2/10] Coarse 3DGS → ${GS_DIR}"
echo "  save_iterations=${SAVE_ARR[*]}  test_iterations=${TEST_ARR[*]}"
lineb_print_paths
echo "=========================================="

python train_gs.py \
    -s "${DATA}" \
    -m "${GS_DIR}" \
    -r 4 --sparse_view_num "${SPARSE_VIEW_NUM}" --sh_degree 2 \
    --init_pcd_name "${INIT_PCD}" \
    --white_background --random_background \
    --iterations "${ITERATIONS}" \
    --save_iterations "${SAVE_ARR[@]}" \
    --test_iterations "${TEST_ARR[@]}" \
    "${DROP_FLAG[@]}"

# 省盘：删 TB / init 副本（不影响 ply 与评测）
rm -f "${GS_DIR}"/events.out.tfevents.* "${GS_DIR}/input.ply" 2>/dev/null || true

echo "[2/10] 完成 → ${GS_DIR}/"
