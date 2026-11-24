#!/bin/bash
# 7. LoRA Fine-Tuning - 微调ControlNet模型
# 这个脚本用于微调 Stable Diffusion和 ControlNet模型，为Gaussian修复阶段做准备

    # --data_dir: 指定数据集路径，包含图像、相机参数等
    # --loo_dir: 指定Leave-One-Out结果路径，包含差异分析数据
    # --exp_name: 指定实验名称，用于保存微调后的模型
    # --prompt: 指定文本提示词，用于描述3D对象
    # --sh_degree: 设置球谐函数的度数，用于颜色表示
    # --resolution: 设置渲染分辨率倍数
    # --sparse_num: 指定稀疏视角数量，这里使用4张图像
    # --bg_white: 使用白色背景进行渲染
    # --sd_locked: 锁定Stable Diffusion的主干网络，只训练LoRA层
    # --train_lora: 启用LoRA训练模式

    # lora 参数
    # --use_prompt_list: 使用提示词列表进行训练
    # --add_diffusion_lora: 为Stable Diffusion添加LoRA层
    # --add_control_lora: 为ControlNet添加LoRA层

CUDA_VISIBLE_DEVICES=2 python train_lora.py \
    --data_dir /data/zhangao_data/3DGS/reconstruction/data/mip360/kitchen \
    --gs_dir output/gs_init/kitchen \
    --loo_dir output/gs_init/kitchen_loo \
    --exp_name controlnet_finetune/kitchen \
    --prompt xxy5syt00 \
    --sh_degree 2 --resolution 4 --sparse_num 9 \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora \
    --add_control_lora \

# 输出：
# output/controlnet_finetune/kitchen/
# ├── ckpts-lora/                       # LoRA模型检查点目录
# │   ├── lora-step=599.ckpt           # 第599步的LoRA权重
# │   ├── lora-step=1199.ckpt          # 第1199步的LoRA权重
# │   └── lora-step=1799.ckpt          # 第1799步的LoRA权重（最终）
# ├── image_log/                        # 图像日志目录
# │   └── train/                       # 训练过程图像
# │       ├── samples_gs-000599_*.png  # 第599步生成的样本图像
# │       ├── samples_gs-001199_*.png  # 第1199步生成的样本图像
# │       ├── samples_gs-001799_*.png  # 第1799步生成的样本图像
# │       ├── denoise_row_*.png        # 去噪过程图像
# │       ├── diffusion_row_*.png      # 扩散过程图像
# │       ├── conditioning_*.png       # 条件输入图像
# │       ├── control_*.png            # ControlNet控制图像
# │       └── reconstruction_*.png     # 重建结果图像
# └── tf_logs/                         # TensorBoard日志目录
#     └── lightning_logs/
#         └── version_0/
#             ├── events.out.tfevents.* # TensorBoard事件日志
#             └── hparams.yaml         # 超参数配置

