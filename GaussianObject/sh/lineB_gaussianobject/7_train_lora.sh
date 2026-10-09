#!/bin/bash
# [7/10] LoRA（对接 GS_DIR / LOO_DIR；略降 lora_rank 省显存）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
lineb_activate

echo "=========================================="
echo "[7/10] Train LoRA → output/${LORA_EXP}"
echo "       gs=${GS_DIR}  loo=${LOO_DIR}  rank=${LORA_RANK}"
echo "=========================================="

python train_lora.py \
    --data_dir "${DATA}" \
    --gs_dir "${GS_DIR}" \
    --loo_dir "${LOO_DIR}" \
    --exp_name "${LORA_EXP}" \
    --prompt xxy5syt00 \
    --sh_degree 2 --resolution 4 --sparse_num "${SPARSE_VIEW_NUM}" \
    --image_size "${LORA_IMAGE_SIZE}" \
    --batch_size 1 \
    --lora_rank "${LORA_RANK}" \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora \
    --add_control_lora

echo "[7/10] 完成 → output/${LORA_EXP}/"
