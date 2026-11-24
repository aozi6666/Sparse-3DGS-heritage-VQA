#!/bin/bash
# [1/10] VGGT + Visual Hull → 初始点云
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

VGGT_CKPT="${VGGT_MODEL}"
# garden 等宽图 images_2≈2594×1681 在 24GB 上会 OOM；可用 VGGT_RESO=4 降采样（与粗训 -r 4 对齐）
VGGT_RESO="${VGGT_RESO:-2}"
lineb_activate

echo "=========================================="
echo "[1/10] VGGT Visual Hull 增强 (reso=${VGGT_RESO})"
echo "=========================================="

python3 ./vggt_visual_hull_enhanced.py \
    --data_dir "${DATA}" \
    --model_path "${VGGT_CKPT}" \
    --sparse_id "${SPARSE_ID}" \
    --reso "${VGGT_RESO}" \
    --voxel_num 100 \
    --vggt_quality_threshold 0.4 \
    --vggt_enhancement_factor 0.2 \
    --not_vis

PLY="${DATA}/${INIT_PCD}.ply"
[[ -f "${PLY}" ]] || { echo "[ERR] missing ${PLY}"; exit 1; }
echo "[1/10] 完成 → ${PLY}"
