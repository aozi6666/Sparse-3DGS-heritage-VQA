#!/bin/bash
# Difix3D增强渲染脚本

# --use_difix_enhancement: 启用Difix3D增强
# --difix_model_path: Difix3D模型路径
# --difix_prompt: 修复提示词
# --difix_strength: 修复强度 (0.0-1.0)
# --difix_guidance_scale: 引导尺度
# --difix_steps: 推理步数

# 设置环境变量
# export PYTHONPATH="${PYTHONPATH}:/data/zhangao_data/3DGS/Difix3D/src"

CUDA_VISIBLE_DEVICES=2 python render_difix.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 9 --sh_degree 2 \
    --init_pcd_name visual_hull_vggt_enhanced_4 \
    --white_background \
    --load_ply output/gaussian_object/kitchen/save/last.ply \
    --skip_train \
    --use_difix_enhancement \
    --difix_model_path "/data/zhangao_data/3DGS/Difix3D/models" \
    --difix_prompt "improve quality" \
    --difix_strength 0.01 \
    --difix_guidance_scale 1.0 \
    --difix_steps 1