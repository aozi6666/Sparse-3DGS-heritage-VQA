#!/bin/bash
# [1/10] VGGT + Visual Hull → 初始点云
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

VGGT_CKPT="${ROOT}/vggt/models/model.pt"
lineb_activate

echo "=========================================="
echo "[1/10] VGGT Visual Hull 增强"
echo "=========================================="

python3 ./vggt_visual_hull_enhanced.py \
    --data_dir "${DATA}" \
    --model_path "${VGGT_CKPT}" \
    --sparse_id "${SPARSE_ID}" \
    --reso 2 \
    --voxel_num 100 \
    --vggt_quality_threshold 0.4 \
    --vggt_enhancement_factor 0.2 \
    --not_vis

PLY="${DATA}/${INIT_PCD}.ply"
[[ -f "${PLY}" ]] || { echo "[ERR] missing ${PLY}"; exit 1; }
echo "[1/10] 完成 → ${PLY}"
