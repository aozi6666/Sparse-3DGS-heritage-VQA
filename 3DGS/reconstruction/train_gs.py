#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
import json
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Optional

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.functional.regression import pearson_corrcoef
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, monodisp
from utils.pose_utils import update_pose, get_loss_tracking
from torch.utils.tensorboard.writer import SummaryWriter
TENSORBOARD_FOUND = True

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    # 如果使用 Dust3r，则使用支持 姿态微调的渲染函数  render_w_pose
    # 否则使用普通的 render 函数
    if args.use_dust3r:
        print('Use pose refinement from dust3r')
        from gaussian_renderer import render_w_pose as render
    else:
        from gaussian_renderer import render

    # 初始化
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, extra_opts=args)
    gaussians.training_setup(opt)

    # 如果指定了断点模型，就加载 .pth 文件并 恢复模型参数
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建 CUDA 事件，测量每一轮训练所耗时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # viewpoint_stack: 相机视角池
    viewpoint_stack, augview_stack = None, None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # 训练主循环
    for iteration in range(first_iter, opt.iterations + 1):   

        # 尝试建立与外部 GUI 的连接（可视化渲染）    
        if network_gui.conn == None:
            network_gui.try_connect()

        # 如果 GUI 连上了，就监听指令，实时渲染画面 或 进行训练控制    
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 记录迭代开始时间，并根据迭代步数更新学习率
        iter_start.record() # type: ignore
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每 1000 步提升一次 SH（球谐函数）的等级，提升细节建模能力
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # 从训练集相机中 随机选择 一个视角 用于本次训练
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        #  Dust3r 的相机姿态优化
        # 如果启用了 Dust3r，就对 旋转 和 平移 分别 创建优化器，用于细致调整相机位姿
        if args.use_dust3r:
            pose_opt_params = [
                {
                    "params": [viewpoint_cam.cam_rot_delta],
                    "lr": 0.003,
                    "name": "rot_{}".format(viewpoint_cam.uid),
                },
                {
                    "params": [viewpoint_cam.cam_trans_delta],
                    "lr": 0.001,
                    "name": "trans_{}".format(viewpoint_cam.uid),
                }
            ]
            pose_optimizer = torch.optim.Adam(pose_opt_params)

        # Render
        # 从某一轮启用调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # 随机背景 或 固定背景
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # 调用渲染函数，得到渲染结果 render_pkg
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        # 调用 cal_loss 函数，计算所有设定的损失（L1、SSIM、轮廓、单目深度）
        loss, Ll1 = cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, tb_writer=tb_writer, iteration=iteration, mono_loss_type=args.mono_loss_type)

        # 反向传播误差，记录迭代结束时间
        loss.backward()
        iter_end.record()  # type: ignore

        # 日志更新 & 模型保存
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            num_gauss = len(gaussians._xyz)
            if iteration % 10 == 0:
                progress_bar.set_postfix({'Loss': f"{ema_loss_for_log:.{7}f}",  'n': f"{num_gauss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            # 点云稠密化与裁剪
            if iteration < opt.densify_until_iter and num_gauss < opt.max_num_splats:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 将重要点进一步细化增加点数（密度提升）
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # 重设透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # 移除离群点（outlier）
                if iteration % opt.remove_outliers_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.remove_outliers(opt, iteration, linear=True)

            # Optimizer step
            # 执行优化器 step
            # 主网络优化器和 Dust3r 相机姿态优化器分别 step() 和 zero_grad()，应用新的相机位姿
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if args.use_dust3r and iteration < opt.pose_iterations:
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none = True)
                    _ = update_pose(viewpoint_cam)

            # 保存 Checkpoint，保存 .pth 断点模型
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt" + str(iteration) + ".pth")

    # Dust3r：保存优化后的相机参数
    if args.use_dust3r:
        with open(os.path.join(dataset.source_path, f'dust3r_{args.sparse_view_num}.json'), 'r') as f:
            json_cameras = json.load(f)
        refined_cameras = []
        for viewpoint_cam, json_camera in zip(scene.getTrainCameras(), json_cameras):
            camera = json_camera
            w2c = np.eye(4)
            w2c[:3, :3] = viewpoint_cam.R.T
            w2c[:3, 3] = viewpoint_cam.T
            c2w = np.linalg.inv(w2c)
            camera['position'] = c2w[:3, 3].tolist()
            camera['rotation'] = c2w[:3, :3].tolist()
            refined_cameras.append(camera)
        with open(os.path.join(scene.model_path, 'refined_cams.json'), 'w') as f:
            json.dump(refined_cameras, f, indent=4)

# 模型输出路径和日志记录
def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
            args.model_path = os.path.join("./output/", unique_str)
        else:
            unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    # 输出路径
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    # 尝试创建 TensorBoard 的日志写入器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
# 在训练中记录日志、执行验证、并输出评估结果
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):

    #  记录训练指标到 TensorBoard
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 创建两个验证配置
        # test：全体测试集相机， train：从训练集相机中抽取少量样本用于训练表现评估（例如：第 5, 10, 15, 20, 25 个）
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        # 遍历上述两个验证配置，如果存在对应的相机列表，则进行评估
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:

                l1_test = 0.0
                psnr_test = 0.0

                # 遍历每个视角（即每个相机）
                for idx, viewpoint in enumerate(config['cameras']):
                    # 渲染函数 renderFunc 渲染该相机对应的图像，并裁剪其像素值在 [0, 1] 区间
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    # 获取该视角对应的原始图像（Ground Truth），同样裁剪像素值并移动到 GPU
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # 如果启用了 TensorBoard，将渲染结果写入 TensorBoard
                    # 计算当前视角下，平均 L1 损失 平均 PSNR 值 累加
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                # 对总损失和 PSNR 求平均，得到该验证集的最终评价指标
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                # 将上面计算的 L1 和 PSNR 写入 TensorBoard，可视化曲线
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 如果启用了 TensorBoard，绘制当前所有高斯点的 不透明度直方图
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()

# 用于计算图像损失的函数
def cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid", tb_writer: Optional[SummaryWriter]=None, iteration=0):
    """
    Calculate the loss of the image, contains l1 loss and ssim loss.
    l1 loss: Ll1 = l1_loss(image, gt_image)
    ssim loss: Lssim = 1 - ssim(image, gt_image)
    Optional: [silhouette loss, monodepth loss]
    该函数计算图像重建的损失，包括 L1 和 SSIM；
    可选地还包含 silhouette（轮廓）和 monodepth（单目深度）损失
    """
    # Ground Truth图像 转换成 渲染图像 相同数据类型
    gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()

    # 如果启用了随机背景，则用 mask 选择前景用 GT 图像、背景用随机背景色
    if opt.random_background:
        gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()
    
    # 计算损失：L1 损失、SSIM 损失、混合 L1 和 SSIM 损失
    Ll1 = l1_loss(image, gt_image)
    Lssim = (1.0 - ssim(image, gt_image))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

    # 如果有 TensorBoard 写入器，则记录 L1 与 SSIM 损失
    if tb_writer is not None:
        tb_writer.add_scalar('loss/l1_loss', Ll1, iteration)
        tb_writer.add_scalar('loss/ssim_loss', Lssim, iteration)

    # 轮廓损失（Silhouette Loss）
    # 如果参数中启用了使用掩膜（mask），开始计算 silhouette（轮廓）损失，BCE（二元交叉熵）、MSE（均方误差）
    if hasattr(args, "use_mask") and args.use_mask:
        if silhouette_loss_type == "bce":
            silhouette_loss = F.binary_cross_entropy(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        elif silhouette_loss_type == "mse":
            silhouette_loss = F.mse_loss(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        else:
            raise NotImplementedError
        loss = loss + opt.lambda_silhouette * silhouette_loss

        # 记录 silhouette 损失到 TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar('loss/silhouette_loss', silhouette_loss, iteration)

    # 单目深度损失（Monocular Depth Loss）
    # 如果视角中有 单目深度图（mono_depth），执行深度损失计算
    if hasattr(viewpoint_cam, "mono_depth") and viewpoint_cam.mono_depth is not None:

        # Case 1: "mid" 中间掩膜匹配
        if mono_loss_type == "mid":
            # we apply masked monocular loss
            gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
            render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
            mask = torch.logical_and(gt_mask, render_mask)
            if mask.sum() < 10:
                depth_loss = 0.0
            else:
                disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]

        # Case 2: "pearson" 皮尔逊相关系数
        elif mono_loss_type == "pearson":
            disp_mono = 1 / viewpoint_cam.mono_depth[viewpoint_cam.mask > 0.5].clamp(1e-6) # shape: [N]
            disp_render = 1 / render_pkg["rendered_depth"][viewpoint_cam.mask > 0.5].clamp(1e-6) # shape: [N]
            depth_loss = (1 - pearson_corrcoef(disp_render, -disp_mono)).mean()

        # Case 3: "dust3r"（逐像素差值 + 时间线性调度）
        elif mono_loss_type == "dust3r":
            gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
            render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
            mask = torch.logical_and(gt_mask, render_mask)
            if mask.sum() < 10:
                depth_loss = 0.0
            else:
                disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                depth_loss = torch.abs((disp_render - disp_mono)).mean()
            depth_loss *= (opt.iterations - iteration) / opt.iterations # linear scheduler
        else:
            raise NotImplementedError
        
        # 汇总深度损失，并写入 TensorBoard
        loss = loss + args.mono_depth_weight * depth_loss
        if tb_writer is not None:
            tb_writer.add_scalar('loss/depth_loss', depth_loss, iteration)
            
    # Dust3r 追踪损失
    if args.use_dust3r:
        image_ab = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
        tracking_loss = get_loss_tracking(image_ab, render_pkg["rendered_alpha"], viewpoint_cam) + args.lambda_t_norm * torch.abs(viewpoint_cam.cam_trans_delta).mean()
        loss = loss + tracking_loss
        if tb_writer is not None:
            tb_writer.add_scalar('loss/tracking_loss', tracking_loss, iteration)

    return loss, Ll1

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    ### some exp args
    # 使用稀疏视角进行训练
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                        help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                        else use dense view. In sparse setting, sparse views will be used as training data, \
                        others will be used as testing data.")
    
    # 是否使用掩码图（遮挡处理）
    parser.add_argument("--use_mask", default=True, help="Use masked image, by default True")

    # 使用 Dust3r 进行姿态优化
    parser.add_argument('--use_dust3r', action='store_true', default=False,
                        help='use dust3r estimated poses')
    parser.add_argument('--dust3r_json', type=str, default=None)
    parser.add_argument("--init_pcd_name", default='origin', type=str, 
                        help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument("--transform_the_world", action="store_true", help="Transform the world to the origin")

    # 单目深度损失权重
    parser.add_argument('--mono_depth_weight', type=float, default=0.0005, help="The rate of monodepth loss")
    parser.add_argument('--lambda_t_norm', type=float, default=0.0005)
    parser.add_argument('--mono_loss_type', type=str, default="mid")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
