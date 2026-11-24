#!/bin/bash
# Leave-One-Out 第二阶段： diffs.pkl - 差异分析结果（核心新增文件）
# 内容: 高斯模型参数变化的统计分析，分析模型在缺失某张图像时的参数变化模式

# -s: 数据路径
# -m: 输出路径
# -r: 分辨率
# --sparse_view_num: 稀疏视角数量
# --sh_degree: 球谐函数阶数
# --init_pcd_name: 初始点云文件名
# --white_background: 白色背景
# --random_background: 随机背景

CUDA_VISIBLE_DEVICES=2  python leave_one_out_stage2.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen_loo \
    -r 4 --sparse_view_num 9 --sh_degree 2 \
    --init_pcd_name visual_hull_vggt_enhanced_4 \
    --white_background --random_background


# 输出：
# output/gs_init/kitchen_loo/
# ├── leave_0/                          # 第1张图像被留出的训练结果
# │   ├── left_image/                   # 渲染图像目录（来自stage1）
# │   │   ├── sample_70000.png
# │   │   ├── sample_80000.png
# │   │   ├── ...
# │   │   └── sample_150000.png
# │   ├── gt.png                       # 被留出图像的真实值（来自stage1）
# │   ├── gaussians_cache.pth          # 高斯模型缓存（来自stage1）
# │   ├── chkpnt6000.pth               # 第6000次迭代的检查点（来自stage1）
# │   ├── diffs.pkl                    # �� 差异分析结果（新增）
# │   ├── point_cloud/                 # 点云模型目录（更新）
# │   │   ├── iteration_70000/
# │   │   │   └── point_cloud.ply
# │   │   └── iteration_150000/
# │   │       └── point_cloud.ply
# │   ├── events.out.tfevents.*        # TensorBoard日志（更新）
# │   ├── cfg_args                     # 配置参数（更新）
# │   └── cameras.json                 # 相机参数
# ├── leave_1/                          # 第2张图像被留出的训练结果
# │   ├── diffs.pkl                    # �� 差异分析结果（新增）
# │   └── ... (其他文件更新)
# ├── leave_2/                          # 第3张图像被留出的训练结果
# │   ├── diffs.pkl                    # �� 差异分析结果（新增）
# │   └── ... (其他文件更新)
# └── leave_3/                          # 第4张图像被留出的训练结果
#     ├── diffs.pkl                    # �� 差异分析结果（新增）
#     └── ... (其他文件更新)