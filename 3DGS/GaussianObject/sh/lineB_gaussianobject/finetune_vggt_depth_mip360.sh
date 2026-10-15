#!/bin/bash
# mip360 multi-scene: decoder-only depth_head finetune on kitchen+bonsai+garden.
# All views (no sparse_9 hold-out). Does NOT overwrite model.pt or depth_head_ft_kitchen.pt.
# Agent does not run this for you — execute locally after reviewing flags.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

RESUME="${RESUME:-${VGGT}/models/model.pt}"
OUT="${OUT:-${VGGT}/models/depth_head_ft_mip360.pt}"
MIP360_ROOT="${MIP360_ROOT:-${GO}/data/mip360}"
SCENES_CSV="${SCENES:-kitchen,bonsai,garden}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-5}"
MAX_SIDE="${MAX_SIDE:-518}"
MIN_VALID="${MIN_VALID:-200}"
USE_MASK_FLAG="${USE_MASK:-1}"

if [[ "$(basename "${OUT}")" == "model.pt" ]]; then
  echo "[ERR] OUT must not be model.pt (got ${OUT})"
  exit 1
fi
if [[ "$(realpath -m "${OUT}")" == "$(realpath -m "${RESUME}")" ]]; then
  echo "[ERR] OUT must differ from RESUME"
  exit 1
fi

IFS=',' read -r -a SCENE_NAMES <<< "${SCENES_CSV}"
DATA_DIRS=()
for name in "${SCENE_NAMES[@]}"; do
  name="$(echo "${name}" | xargs)"
  [[ -n "${name}" ]] || continue
  d="${MIP360_ROOT}/${name}"
  if [[ ! -d "${d}/sparse/0" ]]; then
    echo "[ERR] missing COLMAP sparse/0: ${d}/sparse/0"
    exit 1
  fi
  DATA_DIRS+=("${d}")
done
if [[ "${#DATA_DIRS[@]}" -lt 1 ]]; then
  echo "[ERR] no scenes resolved from SCENES=${SCENES_CSV}"
  exit 1
fi

lineb_activate

MASK_ARGS=()
if [[ "${USE_MASK_FLAG}" == "1" || "${USE_MASK_FLAG}" == "true" || "${USE_MASK_FLAG}" == "True" ]]; then
  MASK_ARGS+=(--use_mask)
fi

echo "=========================================="
echo "[finetune] VGGT depth_head only (mip360 multi-scene, no sparse_9 exclude)"
echo "  RESUME=${RESUME}  (read-only full model)"
echo "  OUT=${OUT}  (depth_head-only, not full VGGT)"
echo "  DATA_DIRS=${DATA_DIRS[*]}"
echo "  EPOCHS=${EPOCHS} LR=${LR} MAX_SIDE=${MAX_SIDE}"
echo "=========================================="

python3 "${GO}/tools/finetune_depth_head_colmap.py" \
  --data_dirs "${DATA_DIRS[@]}" \
  --resume "${RESUME}" \
  --out "${OUT}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --max_side "${MAX_SIDE}" \
  --min_valid "${MIN_VALID}" \
  --batch_size 1 \
  "${MASK_ARGS[@]}"

echo "[finetune] done → ${OUT}"
echo "[finetune] original weights untouched → ${RESUME}"
echo "[finetune] downstream: VGGT_DEPTH_HEAD=${OUT} bash run_depthft_coarse_eval_mip360.sh"
