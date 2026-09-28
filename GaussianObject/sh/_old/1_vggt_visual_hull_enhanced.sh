#!/bin/bash

# VGGT深度增强的视觉外壳生成脚本
# 分阶段设计：基础视觉体 + VGGT深度增强 + 点云合并

# 参数说明：
# --data_dir: 数据目录，包含图像、相机参数、深度图等
# --model_path: VGGT模型路径
# --sparse_id: 稀疏视角ID，对应sparse_4.txt文件
# --reso: 图像分辨率，1=原始尺寸，2=1/2尺寸
# --voxel_num: 体素数量，控制点云密度
# --vggt_quality_threshold: VGGT质量阈值，越低VGGT参与度越高
# --vggt_enhancement_factor: VGGT增强因子，越大增强效果越明显
# --not_vis: 不显示可视化窗口

# 设置GPU
export CUDA_VISIBLE_DEVICES=2

# 执行VGGT深度增强
python3 ./vggt_visual_hull_enhanced.py \
    --data_dir /data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen \
    --model_path /data/zhangao_data/3DGS/vggt/models/model.pt \
    --sparse_id 4 \
    --reso 2 \
    --voxel_num 100 \
    --vggt_quality_threshold 0.4 \
    --vggt_enhancement_factor 0.2 \
    --not_vis

