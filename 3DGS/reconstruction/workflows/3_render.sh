#!/bin/bash
# 3. 渲染测试集结果

    # -m: 指定模型路径，包含训练好的3D高斯模型
    # --sparse_view_num: 指定稀疏视角数量，这里使用4张图像
    # --sh_degree: 设置球谐函数的度数，用于颜色表示
    # --init_pcd_name: 指定初始化点云文件名（不包含.ply扩展名）
    # --white_background: 使用白色背景进行渲染

    # --skip_all: 跳过所有渲染，只进行测试集评估
    # --skip_train: 跳过训练集渲染，只渲染测试集



CUDA_VISIBLE_DEVICES=5 python render.py \
    -m output/gs_init/kitchen_dropgaussian \
    --sparse_view_num 9 --sh_degree 2 \
    --init_pcd_name visual_hull_vggt_enhanced_4 \
    --white_background \
    --skip_train \
    # --skip_all \


# 输出： 
# output/gs_init/kitchen/
# ├── test/
# │   └── ours_10000/                    # 第10000次迭代的测试集渲染结果
# │       ├── renders/                   # 渲染图像目录
# │       │   ├── 00000.png             # 第1张测试图像渲染结果
# │       │   ├── 00001.png             # 第2张测试图像渲染结果
# │       │   ├── ...
# │       │   └── 00034.png             # 第35张测试图像渲染结果
# │       ├── gt/                       # 真实图像目录
# │       │   ├── 00000.png             # 第1张测试图像真实值
# │       │   ├── 00001.png             # 第2张测试图像真实值
# │       │   ├── ...
# │       │   └── 00034.png             # 第35张测试图像真实值
# │       ├── renders.mp4               # 渲染结果视频
# │       ├── gt.mp4                    # 真实图像视频
# │       ├── combined.mp4              # 左右对比视频
# │       ├── depth.mp4                 # 深度图视频
# │       ├── depth_compressed.mp4      # 压缩后的深度视频
# │       └── results.json              # 评估指标结果
# └── ...