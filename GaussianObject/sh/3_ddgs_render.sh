#!/bin/bash
# [3/3] D²GS 渲染 + 评估
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
source /data/zhangao_data/anaconda3/bin/activate GaussianObject
cd /data/zhangao_data/3DGS/DDGS

# 与 2_ddgs_train.sh 的 --output_dir 保持一致
MODEL_DIR="/data/zhangao_data/3DGS/GaussianObject/output/kitchen_ddgs_only/ddgs_model"
DATA_DIR="/data/zhangao_data/3DGS/GaussianObject/output/kitchen_ddgs_only/data"

echo "=========================================="
echo "[3/3] D²GS 渲染"
echo "=========================================="

python render.py \
    -m "${MODEL_DIR}" \
    -s "${DATA_DIR}" \
    --eval \
    -r 8

echo "[3/3] 完成 → ${MODEL_DIR}/test/"
