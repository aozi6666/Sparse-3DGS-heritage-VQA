#!/bin/bash
# 纯 VGGT + depth_head 微调 → visual_hull_vggt_pure_depthft_${SPARSE_ID}.ply
# 不写 classic hull 底；不覆盖 visual_hull_vggt_enhanced_*.ply
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

VGGT_BASE="${VGGT_BASE:-${VGGT}/models/model.pt}"
VGGT_DEPTH_HEAD="${VGGT_DEPTH_HEAD:-${VGGT}/models/depth_head_ft_mip360.pt}"
INIT_PCD="${INIT_PCD:-visual_hull_vggt_pure_depthft_${SPARSE_ID}}"
FORCE_HULL="${FORCE_HULL:-0}"
PURE_STRIDE="${PURE_STRIDE:-4}"
PURE_MAX_POINTS="${PURE_MAX_POINTS:-200000}"

PLY="${DATA}/${INIT_PCD}.ply"
BASELINE_ENHANCED="${DATA}/visual_hull_vggt_enhanced_${SPARSE_ID}.ply"

[[ -f "${VGGT_BASE}" ]] || { echo "[ERR] missing ${VGGT_BASE}"; exit 1; }
[[ -f "${VGGT_DEPTH_HEAD}" ]] || {
  echo "[ERR] missing depth_head: ${VGGT_DEPTH_HEAD}"
  echo "      set VGGT_DEPTH_HEAD=... or run finetune first"
  exit 1
}
if [[ "$(realpath -m "${VGGT_DEPTH_HEAD}")" == "$(realpath -m "${VGGT_BASE}")" ]]; then
  echo "[ERR] VGGT_DEPTH_HEAD must differ from base model.pt"
  exit 1
fi
if [[ "${INIT_PCD}" == "visual_hull_vggt_enhanced_${SPARSE_ID}" ]]; then
  echo "[ERR] INIT_PCD must not be baseline enhanced name"
  exit 1
fi

lineb_activate

echo "=========================================="
echo "[1/pure] VGGT pure + depthft hull"
echo "  VGGT_BASE=${VGGT_BASE}"
echo "  VGGT_DEPTH_HEAD=${VGGT_DEPTH_HEAD}"
echo "  INIT_PCD=${INIT_PCD}"
echo "  PURE_STRIDE=${PURE_STRIDE}  PURE_MAX_POINTS=${PURE_MAX_POINTS}"
echo "=========================================="

if [[ -f "${PLY}" && "${FORCE_HULL}" != "1" ]]; then
  echo "[skip] exists ${PLY} (FORCE_HULL=1 to regenerate)"
  exit 0
fi

# 记录 baseline mtime，结束后核对未被改写
BASE_MTIME=""
if [[ -f "${BASELINE_ENHANCED}" ]]; then
  BASE_MTIME="$(stat -c %Y "${BASELINE_ENHANCED}" 2>/dev/null || true)"
fi

python3 ./vggt_visual_hull_enhanced.py \
  --data_dir "${DATA}" \
  --model_path "${VGGT_DEPTH_HEAD}" \
  --base_model_path "${VGGT_BASE}" \
  --sparse_id "${SPARSE_ID}" \
  --reso 2 \
  --voxel_num 100 \
  --init_mode pure \
  --output_ply_name "${INIT_PCD}" \
  --pure_stride "${PURE_STRIDE}" \
  --pure_max_points "${PURE_MAX_POINTS}" \
  --not_vis

[[ -f "${PLY}" ]] || { echo "[ERR] missing ${PLY}"; exit 1; }

if [[ -n "${BASE_MTIME}" && -f "${BASELINE_ENHANCED}" ]]; then
  NOW="$(stat -c %Y "${BASELINE_ENHANCED}" 2>/dev/null || true)"
  if [[ "${NOW}" != "${BASE_MTIME}" ]]; then
    echo "[ERR] baseline enhanced ply mtime changed — abort"
    exit 1
  fi
fi

echo "[1/pure] 完成 → ${PLY}"
