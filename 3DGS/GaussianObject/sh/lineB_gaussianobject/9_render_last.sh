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
echo "       tag=${REPAIR_TAG}  gs=${GS_DIR}"
echo "=========================================="

# 非默认 REPAIR_TAG（如 kitchen_drop_rv51）时，避免覆盖基线 ours_None 指标
if [[ "${REPAIR_TAG}" != "${GS_NAME}" ]]; then
  for split in test all; do
    src="${GS_DIR}/${split}/ours_None"
    bak="${GS_DIR}/${split}/ours_None__baseline"
    if [[ -d "${src}" && ! -d "${bak}" ]]; then
      echo "[9/10] backup ${src} → ${bak}"
      cp -a "${src}" "${bak}"
    fi
  done
fi

python render.py \
    -m "${GS_DIR}" \
    -s "${DATA}" \
    --sparse_view_num "${SPARSE_VIEW_NUM}" --sh_degree 2 \
    --init_pcd_name "${INIT_PCD}" \
    --white_background \
    --load_ply "${FINAL_PLY}" \
    --skip_train \
    --not_generate_video \
    --not_saveimages

if [[ "${REPAIR_TAG}" != "${GS_NAME}" ]]; then
  for split in test all; do
    src="${GS_DIR}/${split}/ours_None"
    dst="${GS_DIR}/${split}/ours_None__${REPAIR_TAG}"
    if [[ -d "${src}" ]]; then
      echo "[9/10] archive ${src} → ${dst}"
      rm -rf "${dst}"
      cp -a "${src}" "${dst}"
    fi
    # 若有基线备份，恢复 ours_None，避免长期盖掉基线 repair
    bak="${GS_DIR}/${split}/ours_None__baseline"
    if [[ -d "${bak}" ]]; then
      echo "[9/10] restore baseline ${bak} → ${src}"
      rm -rf "${src}"
      cp -a "${bak}" "${src}"
    fi
  done
  echo "[9/10] 完成 → ${GS_DIR}/test/ours_None__${REPAIR_TAG}/results.json"
else
  echo "[9/10] 完成 → ${GS_DIR}/test/*/results.json"
fi
