#!/bin/bash
# 修复训练
# --config: 配置文件
# --train: 训练模式
# --gpu: GPU编号
# tag: 标签
# system.init_dreamer: dreamer初始化路径
# system.exp_name: 实验输出路径
# system.refresh_size: 刷新大小
# data.data_dir: 数据路径
# data.resolution: 分辨率
# data.sparse_num: 稀疏视角数量
# data.prompt: 文本提示
# system.sh_degree: 球谐函数阶数

CUDA_VISIBLE_DEVICES=2 python train_repair.py \
    --config configs/gaussian-object.yaml \
    --train --gpu 2 \
    tag="kitchen" \
    system.init_dreamer="output/gs_init/kitchen" \
    system.exp_name="output/controlnet_finetune/kitchen" \
    system.refresh_size=8 \
    data.data_dir="/data/zhangao_data/3DGS/reconstruction/data/mip360/kitchen" \
    data.resolution=4 \
    data.sparse_num=9 \
    data.prompt="a photo of a xxy5syt00" \
    data.refresh_size=8 \
    system.sh_degree=2

# 输出：

# output/gaussian_object/kitchen/
# ├── save/                            # 模型保存目录（核心输出）
# │   ├── last.ply                    # 🎯 最终修复的3D高斯模型（139MB）
# │   └── controlnet_out/             # ControlNet修复过程图像
# │       ├── it200.png              # 第200次迭代的修复结果
# │       ├── it400.png              # 第400次迭代的修复结果
# │       ├── ...
# │       └── it2600.png             # 第2600次迭代的修复结果
# ├── configs/                        # 配置文件目录
# │   ├── parsed.yaml                # 解析后的配置文件
# │   └── raw.yaml                   # 原始配置文件
# ├── csv_logs/                       # CSV日志目录
# │   └── version_0/
# │       ├── metrics.csv            # 训练指标CSV文件
# │       └── hparams.yaml           # 超参数配置
# ├── tb_logs/                        # TensorBoard日志目录
# │   └── version_0/
# │       ├── checkpoints/           # 模型检查点目录
# │       ├── events.out.tfevents.*  # TensorBoard事件日志
# │       └── hparams.yaml           # 超参数配置
# └── cmd.txt                         # 执行命令记录
