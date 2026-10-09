import argparse
import os
import torch
import pytorch_lightning as pl

from functools import partial
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger

# cldm处理模型结构
from cldm.model import create_model, load_state_dict
from cldm.logger import ImageLogger, LoraCheckpoint
from dataset_lora import GSCacheDataset
from torch.utils.data import DataLoader

# minlora  LoRA 插入逻辑
from minlora import add_lora, LoRAParametrization

# 全局关闭梯度计算
_ = torch.set_grad_enabled(False)


if __name__ == '__main__':

    # 添加命令行参数
    parser = argparse.ArgumentParser(description='Process experiment parameters.')

    parser.add_argument('--model_name', type=str, default='control_v11f1e_sd15_tile')
    parser.add_argument('--sh_degree', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--resolution', type=int, default=4)
    parser.add_argument('--sparse_num', type=int, default=4)
    parser.add_argument('--gs_dir', type=str, default=f'output/gs_init/kitchen')
    parser.add_argument('--data_dir', type=str, default=f'data/mip360/kitchen')
    parser.add_argument('--loo_dir', type=str, default=f'output/gs_init/kitchen_loo')
    parser.add_argument('--prompt', type=str, default='xxy5syt00')
    parser.add_argument('--exp_name', type=str, default=f'controlnet_finetune/kitchen')
    parser.add_argument('--bg_white', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--sd_locked', action='store_true', default=False)
    parser.add_argument('--only_mid_control', action='store_true', default=False)
    parser.add_argument('--train_lora', action='store_true', default=False)
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--callbacks_every_n_train_steps', type=int, default=600)
    parser.add_argument('--max_steps', type=int, default=1800)
    parser.add_argument('--use_prompt_list', action='store_true', default=False)
    parser.add_argument('--manual_noise_reduce_start', type=int, default=100)
    parser.add_argument('--manual_noise_reduce_gamma', type=float, default=0.995)
    parser.add_argument('--cache_max_iter', type=int, default=100)
    parser.add_argument('--add_diffusion_lora', action='store_true', default=False)
    parser.add_argument('--add_control_lora', action='store_true', default=False)
    parser.add_argument('--add_clip_lora', action='store_true', default=False)
    parser.add_argument('--use_dust3r', action='store_true', default=False)

    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据 YAML 配置文件创建 ControlNet 模型
    model = create_model(f'./models/{args.model_name}.yaml').cpu()

    # 加载基础 stable diffusion 模型和 ControlNet 的预训练权重
    model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cpu'), strict=False)
    model.load_state_dict(load_state_dict(f'./models/{args.model_name}.pth', location='cpu'), strict=False)
    
    # 设置模型参数，学习率、是否锁定主模型
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.train_lora = args.train_lora

    # 为不同类型的层（嵌入层、线性层、卷积层）配置 LoRA 插件
    lora_config = {
        nn.Embedding: {
            "weight": partial(LoRAParametrization.from_embedding, rank=args.lora_rank)
        },
        nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=args.lora_rank)
        },
        nn.Conv2d: {
            "weight": partial(LoRAParametrization.from_conv2d, rank=args.lora_rank)
        }
    }

    # 如果启用了 add_diffusion_lora，就在 diffusion 模块中的 transformer 添加 LoRA
    if args.add_diffusion_lora:
        for name, module in model.model.diffusion_model.named_modules():
            if name.endswith('transformer_blocks'):
                add_lora(module, lora_config=lora_config)

    # 控制模块添加 LoRA
    if args.add_control_lora:
        for name, module in model.control_model.named_modules():
            if name.endswith('transformer_blocks'):
                add_lora(module, lora_config=lora_config)
    
    # 在 CLIP 模型（文本编码器）中添加 LoRA
    if args.add_clip_lora:
        add_lora(model.cond_stage_model, lora_config=lora_config)

    # 输出路径 & 数据加载
    exp_path = os.path.join('./output', args.exp_name)
    dataset = GSCacheDataset(
        args.gs_dir, args.data_dir, args.loo_dir,
        prompt=args.prompt,
        bg_white=args.bg_white,
        train=True,
        manual_noise_reduce_gamma=args.manual_noise_reduce_gamma,
        manual_noise_reduce_start=args.manual_noise_reduce_start,
        sh_degree=args.sh_degree,
        image_size=args.image_size,
    resolution=args.resolution, 
        sparse_num=args.sparse_num,
        use_prompt_list=args.use_prompt_list,
        cache_max_iter=args.cache_max_iter,
        use_dust3r=args.use_dust3r,
    )

    # 加载自定义的数据集，包括高斯球信息、视角图像
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)

    # TensorBoard 用于记录训练过程
    loggers = [
        TensorBoardLogger(os.path.join(exp_path, 'tf_logs'))
    ]

    # ImageLogger: 每隔一段时间保存生成图像
    # LoraCheckpoint: 保存 LoRA 层权重
    callbacks = [
        ImageLogger(exp_dir=exp_path, every_n_train_steps=args.callbacks_every_n_train_steps, \
                    log_images_kwargs = {"plot_diffusion_rows": True, "sample": True}),
        LoraCheckpoint(exp_dir=exp_path, every_n_train_steps=args.callbacks_every_n_train_steps)
    ]

    # 使用 PyTorch Lightning 的 Trainer 启动训练
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=32,
        logger=loggers,
        callbacks=callbacks,
        max_steps=args.max_steps,
        check_val_every_n_epoch=args.callbacks_every_n_train_steps//len(dataset)*2
    )

    trainer.fit(model, dataloader)
