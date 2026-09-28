#!/bin/bash
# D²GS渲染脚本
# 专门用于渲染D²GS训练好的模型

echo "=========================================="
echo "D²GS模型渲染脚本"
echo "=========================================="

# 设置GPU
export CUDA_VISIBLE_DEVICES=3

# 激活环境
source /data/zhangao_data/anaconda3/bin/activate GaussianObject

# 切换到D²GS目录
cd /data/zhangao_data/3DGS/DDGS

# D²GS渲染命令
python render.py \
    -m /data/zhangao_data/3DGS/GaussianObject/output/kitchen_ddgs_success/ddgs_model \
    -s /data/zhangao_data/3DGS/GaussianObject/output/kitchen_ddgs_success/data \
    --eval \
    -r 8

echo "=========================================="
echo "D²GS渲染完成！"
echo "=========================================="
echo "渲染结果位置:"
echo "  - 渲染图像: /data/zhangao_data/3DGS/GaussianObject/output/kitchen_ddgs_success/ddgs_model/test/ours_1000/renders/"
echo "  - 真实图像: /data/zhangao_data/3DGS/GaussianObject/output/kitchen_ddgs_success/ddgs_model/test/ours_1000/gt/"
echo "  - 评估结果: /data/zhangao_data/3DGS/GaussianObject/output/kitchen_ddgs_success/ddgs_model/test/ours_1000/results.json"
echo "=========================================="
