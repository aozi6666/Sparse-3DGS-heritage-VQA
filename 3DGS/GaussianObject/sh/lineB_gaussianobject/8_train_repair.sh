#!/bin/bash
# [8/10] Gaussian repair（init = Drop/粗训 GS_DIR）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
GPU="${CUDA_VISIBLE_DEVICES}"
lineb_activate

echo "=========================================="
echo "[8/10] Train repair  tag=${REPAIR_TAG}"
echo "       init=${GS_DIR}  controlnet=output/${LORA_EXP}"
echo "=========================================="

python train_repair.py \
    --config configs/gaussian-object.yaml \
    --train --gpu "${GPU}" \
    tag="${REPAIR_TAG}" \
    system.init_dreamer="${GS_DIR}" \
    system.exp_name="output/${LORA_EXP}" \
    system.refresh_size=8 \
    data.data_dir="${DATA}" \
    data.resolution=4 \
    data.sparse_num="${SPARSE_VIEW_NUM}" \
    data.prompt="a photo of a xxy5syt00" \
    data.refresh_size=8 \
    system.sh_degree=2

echo "[8/10] 完成 → ${FINAL_PLY}"
