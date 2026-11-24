#!/bin/bash
# 同 hull init ± Drop 对照（仅粗训+评测）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SPARSE_ID="${SPARSE_ID:-4}"
export SPARSE_VIEW_NUM="${SPARSE_VIEW_NUM:-9}"

echo "=========================================="
echo "Drop ablation: USE_DROP=0 then 1"
echo "=========================================="

USE_DROP=0 bash "${SCRIPT_DIR}/2_train_gs.sh"
USE_DROP=0 bash "${SCRIPT_DIR}/3_render.sh"

USE_DROP=1 bash "${SCRIPT_DIR}/2_train_gs.sh"
USE_DROP=1 bash "${SCRIPT_DIR}/3_render.sh"

echo "=========================================="
echo "Ablation done."
echo "  no Drop → output/gs_init/kitchen/"
echo "  Drop    → output/gs_init/kitchen_drop/"
echo "=========================================="
