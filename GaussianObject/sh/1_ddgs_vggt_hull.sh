#!/bin/bash
# [1/3] VGGT + Visual Hull 深度增强 → 初始点云
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
source /data/zhangao_data/anaconda3/bin/activate GaussianObject
cd /data/zhangao_data/3DGS/GaussianObject

echo "=========================================="
echo "[1/3] VGGT Visual Hull 增强"
echo "=========================================="

python3 ./vggt_visual_hull_enhanced.py \
    --data_dir /data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen \
    --model_path /data/zhangao_data/3DGS/vggt/models/model.pt \
    --sparse_id 4 \
    --reso 2 \
    --voxel_num 100 \
    --vggt_quality_threshold 0.4 \
    --vggt_enhancement_factor 0.2 \
    --not_vis

echo "[1/3] 完成 → 检查 good_visual_hull_vggt_enhanced_4.ply"
