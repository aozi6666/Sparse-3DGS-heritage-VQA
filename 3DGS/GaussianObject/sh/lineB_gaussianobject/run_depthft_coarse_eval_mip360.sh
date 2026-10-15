#!/bin/bash
# Depthft coarse eval with mip360 multi-scene depth_head weights.
# New GS dir kitchen_drop_depthft_mip360 — does not touch kitchen_drop or kitchen_drop_depthft.
#
# Usage:
#   bash sh/lineB_gaussianobject/run_depthft_coarse_eval_mip360.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

export VGGT_DEPTH_HEAD="${VGGT_DEPTH_HEAD:-${VGGT}/models/depth_head_ft_mip360.pt}"
export GS_NAME="${GS_NAME:-kitchen_drop_depthft_mip360}"
export SPARSE_ID="${SPARSE_ID:-4}"
export INIT_PCD="${INIT_PCD:-visual_hull_vggt_depthft_mip360_${SPARSE_ID}}"
export USE_DROP="${USE_DROP:-1}"
export SKIP_PATH="${SKIP_PATH:-1}"
export GS_DIR="output/gs_init/${GS_NAME}"
export LOO_DIR="output/gs_init/${GS_NAME}_loo"

if [[ ! -f "${VGGT_DEPTH_HEAD}" ]]; then
  echo "[ERR] missing depth_head: ${VGGT_DEPTH_HEAD}"
  echo "      Run finetune_vggt_depth_mip360.sh first."
  exit 1
fi

echo "=========================================="
echo "mip360 depthft coarse eval wrapper"
echo "  VGGT_DEPTH_HEAD=${VGGT_DEPTH_HEAD}"
echo "  GS_NAME=${GS_NAME}  INIT_PCD=${INIT_PCD}"
echo "=========================================="

bash "${SCRIPT_DIR}/run_depthft_coarse_eval.sh"
