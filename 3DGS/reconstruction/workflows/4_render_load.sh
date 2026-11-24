#!/bin/bash
# 4. 渲染路径 - 生成相机轨迹视频
# 这个脚本用于渲染自定义相机路径，生成围绕3D对象的动态视频

    # -m: 指定模型路径，包含训练好的3D高斯模型
    # --sparse_view_num: 指定稀疏视角数量，这里使用4张图像
    # --sh_degree: 设置球谐函数的度数，用于颜色表示
    # --init_pcd_name: 指定初始化点云文件名（不包含.ply扩展名）
    # --white_background: 使用白色背景进行渲染
    # --render_path: 启用路径渲染模式，生成相机轨迹视频
    
# 设置使用的GPU设备
CUDA_VISIBLE_DEVICES=2 python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 9 \
    --sh_degree 2 \
    --init_pcd_name visual_hull_vggt_enhanced_4 \
    --white_background \
    --render_path

# 输出目录结构：
# output/gs_init/kitchen/
# ├── render/                           # 路径渲染结果目录
# │   └── ours_10000/                   # 第10000次迭代的路径渲染结果
# │       ├── renders/                  # 路径渲染图像目录
# │       │   ├── 00000.png            # 第1帧路径渲染结果
# │       │   ├── 00001.png            # 第2帧路径渲染结果
# │       │   ├── ...
# │       │   └── 00119.png            # 第120帧路径渲染结果
# │       ├── renders.mp4              # 路径渲染视频
# │       ├── depth.mp4                # 路径深度图视频
# │       └── depth_compressed.mp4     # 压缩后的深度视频
# └── ...