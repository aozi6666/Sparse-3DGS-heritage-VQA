#!/bin/bash
# 2.训练粗略的 3DGS 模型，用于初始化3D高斯 (集成DropGaussian)

    # -s: 指定数据集的路径，包含图像、相机参数等
    # -m: 指定模型保存的路径，训练好的模型会保存在这里
    # -r: 设置渲染分辨率倍数，4表示4倍分辨率
    # --sparse_view_num: 指定使用的稀疏视图数量，这里使用9张图像
    # --sh_degree: 设置球谐函数的度数，用于表示颜色的方向性
    # --init_pcd_name: 指定初始化点云文件名，用于初始化3D高斯
    # --white_background: 使用白色背景进行渲染
    # --random_background: 使用随机背景颜色，增加训练多样性
    # --use_dropgaussian: 启用DropGaussian正则化
    # --drop_rate: DropGaussian的dropout率 (0.0-1.0)
    # --drop_schedule: DropGaussian调度策略 (linear/cosine/constant)

CUDA_VISIBLE_DEVICES=5 python train_gs_dropgaussian.py \
    -s /data/zhangao_data/3DGS/reconstruction/data/mip360/kitchen \
    -m output/gs_init/kitchen_dropgaussian \
    -r 4 --sparse_view_num 9 --sh_degree 2 \
    --init_pcd_name visual_hull_vggt_enhanced_4  \
    --white_background --random_background \
    --use_dropgaussian --drop_rate 0.1 --drop_schedule linear

# 输出： 
# """output/gs_init/kitchen_dropgaussian/
# ├── events.out.tfevents.1757039468.A100-205.1609775.0  # TensorBoard日志文件
# ├── point_cloud/                                        # 点云模型目录
# │   ├── iteration_7000/
# │   │   └── point_cloud.ply                            # 第7000次迭代的3D高斯模型
# │   └── iteration_10000/
# │       └── point_cloud.ply                            # 第10000次迭代的3D高斯模型
# ├── input.ply                                          # 输入初始化点云
# ├── cameras.json                                       # 相机参数文件
# └── cfg_args                                           # 训练配置参数
# """