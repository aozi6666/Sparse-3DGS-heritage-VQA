#!/bin/bash
# [2/3] D²GS 训练（点云 → COLMAP/掩码 → DDGS train）
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
source /data/zhangao_data/anaconda3/bin/activate GaussianObject
cd /data/zhangao_data/3DGS/GaussianObject

OUT_DIR="./output/kitchen_ddgs_only"

echo "=========================================="
echo "[2/3] D²GS 训练"
echo "=========================================="

python ddgs_training_from_vggt.py \
    --vggt_output_dir /data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen \
    --data_dir /data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen \
    --sparse_id 4 \
    --resolution 1 \
    --pointcloud_file "/data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen/good_visual_hull_vggt_enhanced_4.ply" \
    --output_dir "${OUT_DIR}" \
    --ddgs_iterations 30000 \
    --depth_weight 0.1 \
    --density_weight 0.9 \
    --drop_min 0.05 \
    --drop_max 0.5 \
    --mask_param 10 \
    --lambda_far 0.5 \
    --n_views 12 \
    --resolution_factor 8

echo "[2/3] 完成 → ${OUT_DIR}/ddgs_model/"
