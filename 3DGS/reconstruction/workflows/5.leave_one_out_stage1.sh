#!/bin/bash
# Leave-One-Out 第一阶段
# -s: 数据路径
# -m: 输出路径
# -r: 分辨率
# --sparse_view_num: 稀疏视角数量
# --sh_degree: 球谐函数阶数
# --init_pcd_name: 初始点云文件名
# --white_background: 白色背景
# --random_background: 随机背景

CUDA_VISIBLE_DEVICES=2  python leave_one_out_stage1.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen_loo \
    -r 4 --sparse_view_num 9 --sh_degree 2 \
    --init_pcd_name visual_hull_vggt_enhanced_4 \
    --white_background --random_background


# 输出: 
# output/gs_init/kitchen_loo/
# ├── leave_0/                          # 第1张图像被留出的训练结果
# │   ├── left_image/                   # 渲染图像目录
# │   │   ├── sample_70000.png         # 第70000次迭代的渲染结果
# │   │   ├── sample_80000.png         # 第80000次迭代的渲染结果
# │   │   ├── ...
# │   │   └── sample_150000.png        # 第150000次迭代的渲染结果
# │   ├── gt.png                       # 被留出图像的真实值
# │   ├── gaussians_cache.pth          # 高斯模型缓存
# │   ├── point_cloud/                 # 点云模型目录
# │   │   ├── iteration_70000/
# │   │   │   └── point_cloud.ply      # 第70000次迭代的模型
# │   │   └── iteration_150000/
# │   │       └── point_cloud.ply      # 第150000次迭代的模型
# │   ├── events.out.tfevents.*        # TensorBoard日志
# │   ├── cfg_args                     # 配置参数
# │   └── cameras.json                 # 相机参数
# ├── leave_1/                          # 第2张图像被留出的训练结果
# │   ├── left_image/
# │   │   ├── sample_70000.png
# │   │   ├── ...
# │   │   └── sample_150000.png
# │   ├── gt.png
# │   ├── gaussians_cache.pth
# │   └── ...
# ├── leave_2/                          # 第3张图像被留出的训练结果
# │   └── ...
# └── leave_3/                          # 第4张图像被留出的训练结果
#     └── ...
