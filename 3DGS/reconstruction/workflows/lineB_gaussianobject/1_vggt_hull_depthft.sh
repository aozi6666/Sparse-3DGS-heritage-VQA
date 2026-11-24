#!/bin/bash
# [1/10] VGGT Visual Hull with finetuned depth_head (base model.pt + overlay).
# Does not modify 1_vggt_hull.sh. Protects baseline enhanced ply via backup/restore.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

# Base full weights + depth_head-only finetune artifact
VGGT_BASE="${VGGT_BASE:-${VGGT}/models/model.pt}"
# Prefer new name; accept legacy VGGT_MODEL / model_depthft_* if user exports them
if [[ -n "${VGGT_DEPTH_HEAD:-}" ]]; then
  :
elif [[ -n "${VGGT_MODEL:-}" && "$(realpath -m "${VGGT_MODEL}")" != "$(realpath -m "${VGGT_BASE}")" ]]; then
  VGGT_DEPTH_HEAD="${VGGT_MODEL}"
else
  VGGT_DEPTH_HEAD="${VGGT}/models/depth_head_ft_kitchen.pt"
fi
INIT_PCD="${INIT_PCD:-visual_hull_vggt_depthft_${SPARSE_ID}}"

BASE_PLY="${DATA}/visual_hull_vggt_enhanced_${SPARSE_ID}.ply"
DEPTHFT_PLY="${DATA}/${INIT_PCD}.ply"
BAK_PLY="${BASE_PLY}.bak_before_depthft"

if [[ ! -f "${VGGT_BASE}" ]]; then
  echo "[ERR] missing base VGGT weights: ${VGGT_BASE}"
  exit 1
fi
if [[ ! -f "${VGGT_DEPTH_HEAD}" ]]; then
  echo "[ERR] missing depth_head finetune: ${VGGT_DEPTH_HEAD}"
  echo "      Run finetune_vggt_depth_kitchen.sh first (or set VGGT_DEPTH_HEAD=...)."
  exit 1
fi
if [[ "$(realpath -m "${VGGT_DEPTH_HEAD}")" == "$(realpath -m "${VGGT_BASE}")" ]]; then
  echo "[ERR] VGGT_DEPTH_HEAD must differ from base model.pt"
  exit 1
fi

VGGT_RESO="${VGGT_RESO:-2}"
lineb_activate

echo "=========================================="
echo "[1/10] VGGT Visual Hull (depthft) reso=${VGGT_RESO}"
echo "  VGGT_BASE=${VGGT_BASE}"
echo "  VGGT_DEPTH_HEAD=${VGGT_DEPTH_HEAD}"
echo "  INIT_PCD=${INIT_PCD}"
echo "  DATA=${DATA}"
echo "=========================================="

# Hull script always writes visual_hull_vggt_enhanced_${SPARSE_ID}.ply
if [[ -f "${BASE_PLY}" ]]; then
  cp -a "${BASE_PLY}" "${BAK_PLY}"
  echo "[info] backed up baseline → ${BAK_PLY}"
fi

cleanup_restore() {
  local rc=$?
  if [[ -f "${BAK_PLY}" ]]; then
    mv -f "${BAK_PLY}" "${BASE_PLY}"
    echo "[info] restored baseline ply → ${BASE_PLY}"
  fi
  exit "${rc}"
}
trap cleanup_restore EXIT

# model_path = depth_head-only file; base_model_path = full model.pt
python3 ./vggt_visual_hull_enhanced.py \
    --data_dir "${DATA}" \
    --model_path "${VGGT_DEPTH_HEAD}" \
    --base_model_path "${VGGT_BASE}" \
    --sparse_id "${SPARSE_ID}" \
    --reso "${VGGT_RESO}" \
    --voxel_num 100 \
    --vggt_quality_threshold 0.4 \
    --vggt_enhancement_factor 0.2 \
    --not_vis

[[ -f "${BASE_PLY}" ]] || { echo "[ERR] hull did not produce ${BASE_PLY}"; exit 1; }
mv -f "${BASE_PLY}" "${DEPTHFT_PLY}"
echo "[1/10] depthft hull → ${DEPTHFT_PLY}"

# Restore baseline before trap exits successfully
if [[ -f "${BAK_PLY}" ]]; then
  mv -f "${BAK_PLY}" "${BASE_PLY}"
  echo "[info] restored baseline ply → ${BASE_PLY}"
fi
trap - EXIT

[[ -f "${DEPTHFT_PLY}" ]] || { echo "[ERR] missing ${DEPTHFT_PLY}"; exit 1; }
echo "[1/10] 完成 → ${DEPTHFT_PLY} (baseline ${BASE_PLY} unchanged if it existed)"
