#!/bin/bash
# [3/3] D²GS 渲染 + 评估 + 最终视频
# 输出：
#   ${MODEL_DIR}/test/ours_<iter>/renders/*.png   # 最终渲染图（保留）
#   ${MODEL_DIR}/test/ours_<iter>/gt/*.png
#   ${MODEL_DIR}/test/ours_<iter>/renders.mp4
#   ${MODEL_DIR}/test/ours_<iter>/gt.mp4
#   ${MODEL_DIR}/test/ours_<iter>/combined.mp4    # 左右对比视频（主看这个）
#   ${MODEL_DIR}/metrics_<iter>.txt
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
source /data/zhangao_data/anaconda3/bin/activate GaussianObject
cd /data/zhangao_data/3DGS/DDGS

# 与 2_ddgs_train.sh 的 --output_dir 保持一致
MODEL_DIR="/data/zhangao_data/3DGS/GaussianObject/output/kitchen_ddgs_only/ddgs_model"
DATA_DIR="/data/zhangao_data/3DGS/GaussianObject/output/kitchen_ddgs_only/data"

echo "=========================================="
echo "[3/3] D²GS 渲染 + 指标 + 视频"
echo "=========================================="

python render.py \
    -m "${MODEL_DIR}" \
    -s "${DATA_DIR}" \
    --eval \
    -r 8 \
    --skip_train

echo "[3/3] 完成"
echo "  渲染图 → ${MODEL_DIR}/test/ours_*/renders/"
echo "  指标   → ${MODEL_DIR}/metrics_*.txt"
echo "  视频   → ${MODEL_DIR}/test/ours_*/combined.mp4"
