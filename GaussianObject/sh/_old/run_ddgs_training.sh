#!/bin/bash

# D²GS训练专用脚本
# 基于VGGT + Visual Hull + 深度增强的结果进行D²GS训练

# 设置GPU
export CUDA_VISIBLE_DEVICES=3

# 激活环境
source /data/zhangao_data/anaconda3/bin/activate GaussianObject

# 进入项目目录
cd /data/zhangao_data/3DGS/GaussianObject

echo "=========================================="
echo "D²GS训练专用脚本"
echo "基于VGGT + Visual Hull + 深度增强的结果"
echo "=========================================="

# 执行D²GS训练
python ddgs_training_from_vggt.py \
    --vggt_output_dir /data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen \
    --data_dir /data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen \
    --sparse_id 4 \
    --resolution 1 \
    --pointcloud_file "/data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen/good_visual_hull_vggt_enhanced_4.ply" \
    --output_dir ./output/kitchen_ddgs_only \
    --ddgs_iterations 30000 \
    --depth_weight 0.1 \
    --density_weight 0.9 \
    --drop_min 0.05 \
    --drop_max 0.5 \
    --mask_param 10 \
    --lambda_far 0.5 \
    --n_views 12 \
    --resolution_factor 8 \
    

echo "=========================================="
echo "D²GS训练流程执行完成！"
echo "=========================================="
echo "输出文件:"
echo "  - COLMAP结果: ./output/kitchen_ddgs_only/colmap_output/"
echo "  - D²GS模型: ./output/kitchen_ddgs_only/ddgs_model/"
echo "  - 渲染结果: ./output/kitchen_ddgs_only/ddgs_model/renders/"
echo "=========================================="