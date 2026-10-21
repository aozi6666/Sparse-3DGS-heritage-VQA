#!/bin/bash
# Kitchen pilot: decoder-only depth_head finetune on COLMAP sparse depth.
# Does NOT overwrite 3DGS/vggt/models/model.pt
# Agent does not run this for you — execute locally after reviewing flags.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

RESUME="${RESUME:-${VGGT}/models/model.pt}"
# depth_head-only artifact (not a full VGGT copy); hull merges with RESUME/model.pt
OUT="${OUT:-${VGGT}/models/depth_head_ft_kitchen.pt}"
EXCLUDE_SPARSE="${EXCLUDE_SPARSE:-${DATA}/sparse_9.txt}"
EPOCHS="${EPOCHS:-3}"
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

lineb_activate

MASK_ARGS=()
if [[ "${USE_MASK_FLAG}" == "1" || "${USE_MASK_FLAG}" == "true" || "${USE_MASK_FLAG}" == "True" ]]; then
  MASK_ARGS+=(--use_mask)
fi

echo "=========================================="
echo "[finetune] VGGT depth_head only (kitchen pilot)"
echo "  RESUME=${RESUME}  (read-only full model)"
echo "  OUT=${OUT}  (depth_head-only, not full VGGT)"
echo "  DATA=${DATA}"
echo "  EXCLUDE_SPARSE=${EXCLUDE_SPARSE}"
echo "  EPOCHS=${EPOCHS} LR=${LR} MAX_SIDE=${MAX_SIDE}"
echo "=========================================="

python3 "${GO}/tools/finetune_depth_head_colmap.py" \
  --data_dir "${DATA}" \
  --resume "${RESUME}" \
  --out "${OUT}" \
  --exclude_sparse "${EXCLUDE_SPARSE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --max_side "${MAX_SIDE}" \
  --min_valid "${MIN_VALID}" \
  --batch_size 1 \
  "${MASK_ARGS[@]}"

echo "[finetune] done → ${OUT}"
echo "[finetune] original weights untouched → ${RESUME}"
