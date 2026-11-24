#!/bin/bash
# [8/10] Gaussian repair（init = Drop/粗训 GS_DIR）
# CADR 相关 env（可选）：
#   USE_CONF_WEIGHT=0|1
#   CADR_TRUST_ONLY=1          → system.conf_trust_only (A2)
#   USE_COV_IN_NEED=1          → system.use_cov_in_need (A4)
#   USE_ADAPTIVE_DENOISE=1     → system.use_adaptive_denoise (A5)
#   SAVE_CONTROLNET_PREVIEW=1  → 写 controlnet_out/it*.png（默认关，省盘）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
GPU="${CUDA_VISIBLE_DEVICES}"
lineb_activate

bool_from_env() {
  local v="${1:-0}"
  if [[ "${v}" == "1" || "${v}" == "True" || "${v}" == "true" ]]; then
    echo True
  else
    echo False
  fi
}

CONF_WEIGHT_BOOL="$(bool_from_env "${USE_CONF_WEIGHT}")"
TRUST_ONLY_BOOL="$(bool_from_env "${CADR_TRUST_ONLY:-0}")"
COV_NEED_BOOL="$(bool_from_env "${USE_COV_IN_NEED:-0}")"
ADEN_BOOL="$(bool_from_env "${USE_ADAPTIVE_DENOISE:-0}")"
PREVIEW_BOOL="$(bool_from_env "${SAVE_CONTROLNET_PREVIEW:-0}")"

echo "=========================================="
echo "[8/10] Train repair  tag=${REPAIR_TAG}"
echo "       init=${GS_DIR}  controlnet=output/${LORA_EXP}"
echo "       sd_ckpt=${SD_CKPT}  use_conf_weight=${CONF_WEIGHT_BOOL}"
echo "       conf_trust_only=${TRUST_ONLY_BOOL}  use_cov_in_need=${COV_NEED_BOOL}"
echo "       use_adaptive_denoise=${ADEN_BOOL}  save_controlnet_preview=${PREVIEW_BOOL}"
echo "=========================================="

# lora_rank / add_clip_lora 必须与 7_train_lora.sh 一致（默认 LORA_RANK=32，未训 clip LoRA）
# sd_ckpt 必须与训 LoRA 时同一底模
python train_repair.py \
    --config configs/gaussian-object.yaml \
    --train --gpu "${GPU}" \
    tag="${REPAIR_TAG}" \
    system.init_dreamer="${GS_DIR}" \
    system.exp_name="output/${LORA_EXP}" \
    system.sd_ckpt="${SD_CKPT}" \
    system.lora_rank="${LORA_RANK}" \
    system.add_clip_lora=False \
    system.refresh_size=8 \
    system.use_conf_weight="${CONF_WEIGHT_BOOL}" \
    system.conf_trust_only="${TRUST_ONLY_BOOL}" \
    system.use_cov_in_need="${COV_NEED_BOOL}" \
    system.use_adaptive_denoise="${ADEN_BOOL}" \
    system.save_controlnet_preview="${PREVIEW_BOOL}" \
    data.data_dir="${DATA}" \
    data.resolution=4 \
    data.sparse_num="${SPARSE_VIEW_NUM}" \
    data.prompt="a photo of a xxy5syt00" \
    data.refresh_size=8 \
    system.sh_degree=2

echo "[8/10] 完成 → ${FINAL_PLY}"
