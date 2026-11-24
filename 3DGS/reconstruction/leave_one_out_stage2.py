import sys
import os
import uuid
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from random import randint

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision
from tqdm import tqdm
from torchmetrics.functional.regression import pearson_corrcoef

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, monodisp
from utils.image_utils import psnr
from gaussian_renderer import render
from scene import Scene, GaussianModel


try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def leave_one_out_training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, train_id):
    first_iter = 6000 # in this code, we just use the data from 6000 iter 从第6000步开始训练

    # 初始化日志记录器，用于输出TensorBoard日志
    tb_writer = prepare_output_and_logger(dataset)  

    # 初始化高斯模型，场景对象
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False, extra_opts=args) # make sure we load "densify_until_iter" model
    gaussians.training_setup(opt)  # 优化器参数

    # 模型检查点
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    #  背景颜色设置
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    #  CUDA计时器
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    #  初始化变量，train_id 是当前“留出的”视角（用于测试）
    num_id, image_id = train_id
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    # 进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    #  主训练循环
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record() # type: ignore

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每10000步提高球谐展开的等级
        if iteration % 10000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera 选取训练视角
        # 留下一个视角用于测试，其他用于训练
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()[:num_id] + scene.getTrainCameras().copy()[num_id+1:] # leave one out
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render 渲染和调试
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss 损失 L1、SSIM
        loss, Ll1 = cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, mono_loss_type=args.mono_loss_type, iteration=iteration)

        loss.backward()

        iter_end.record()  # type: ignore

        with torch.no_grad():
            # Progress bar 记录时间、日志和可视化
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            num_gauss = len(gaussians._xyz)
            if iteration % 10 == 0:
                progress_bar.set_postfix({'Loss': f"{ema_loss_for_log:.{7}f}",  'n': f"{num_gauss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log 训练日志记录
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # Save 保存模型
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification 稠密化操作
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # # 根据 可视性、半径 等特征进行密度调整和 裁剪（prune）
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # 定期重置透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # if iteration % (opt.opacity_reset_interval // 2) == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                # 移除离群点（outliers）
                if iteration % opt.remove_outliers_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.remove_outliers(opt, iteration, linear=True)

            # Optimizer step 优化器更新
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # in the end, we use the cached gaussians and the final gaussians to get the \delta gaussians
    # 最终差异分析，记录位置、特征、透明度、缩放、旋转等的均值和标准差
    cur_status = gaussians.cache
    pre_status = torch.load(os.path.join(args.model_path, 'gaussians_cache.pth'))
    diffs = {}
    keys = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']
    for key, pre_c, cur_c in zip(keys, pre_status, cur_status):
        diff = pre_c - cur_c
        mean_diff = torch.mean(diff, dim=0).cpu().numpy()
        std_diff = torch.std(diff, dim=0).cpu().numpy()
        diffs[key] = [mean_diff, std_diff]

    # 比较结果保存为 pickle 文件
    import pickle
    with open(os.path.join(args.model_path, 'diffs.pkl'), 'wb') as f:
        pickle.dump(diffs, f)

    return dataset, gaussians, scene

# 输出目录和日志 TensorBoard
def prepare_output_and_logger(args):    

    # 指定模型输出路径
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
            args.model_path = os.path.join("./output/", unique_str)
        else:
            unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder 创建目录并保存配置
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer  初始化 TensorBoard 日志器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# 记录训练进度与测试评估
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    
    # 写入训练损失到 TensorBoard
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)  # elapsed: 每次迭代所需时间

    # Report test and samples of training set
    # 如果当前是测试轮次，执行评估
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        # 执行测试评估
        """测试集：全体 TestCameras  训练集：抽样几个 TrainCameras 进行可视化评估"""
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # 对每个视角执行渲染 + 评价
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                # 对每个视角进行渲染，并获取对应的 ground truth 图像
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    #  记录可视化结果
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                   
                    # 计算评估指标,平均 L1 和 PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])      

                # 输出并记录评估结果    
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        # 额外记录模型信息,当前所有点的透明度直方图 点云中高斯点的数量
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# 计算图像损失函数
def cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid", iteration=0):
    """
    Calculate the loss of the image, contains l1 loss and ssim loss.
    l1 loss: Ll1 = l1_loss(image, gt_image)
    ssim loss: Lssim = 1 - ssim(image, gt_image)

    Optional: [silhouette loss, monodepth loss]
    silhouette_loss_type: 轮廓损失类型（默认为 "bce"）
    mono_loss_type: 单目深度损失类型（默认为 "mid"）
    """

    # 获取地面真实图像
    gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()

    # 如果启用了随机背景，将地面真实图像和背景结合
    if opt.random_background:
        gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

    # 处理轮廓损失（可选） (如果设置了 use_mask)
    if hasattr(args, "use_mask") and args.use_mask:

        # BCE 损失：使用 二元交叉熵 来计算轮廓损失
        if silhouette_loss_type == "bce":
            silhouette_loss = F.binary_cross_entropy(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        # MSE 损失：使用均方误差来计算轮廓损失
        elif silhouette_loss_type == "mse":
            silhouette_loss = F.mse_loss(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        else:
            raise NotImplementedError
        # 计算得到的轮廓损失加到总损失中
        loss = loss + opt.lambda_silhouette * silhouette_loss
    
    # 计算单目深度损失（可选）
    if hasattr(viewpoint_cam, "mono_depth") and  viewpoint_cam.mono_depth is not None:

        # 选择使用 "mid" 类型
        if mono_loss_type == "mid":
            # we apply masked monocular loss

            # 创建掩码,表示地面 真实图像 和 渲染图像 的有效区域
            gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
            render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
            mask = torch.logical_and(gt_mask, render_mask)

            # 使用 L1 损失 计算 单目深度 和 渲染深度 之间的差异
            if mask.sum() < 10:
                depth_loss = 0.0
            else:
                disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]

        # 选择 "pearson" 类型，计算 皮尔逊相关系数
        elif mono_loss_type == "pearson":
            zoe_depth = viewpoint_cam.mono_depth[viewpoint_cam.mask > 0.5].clamp(1e-6)
            rendered_depth = render_pkg["rendered_depth"][viewpoint_cam.mask > 0.5].clamp(1e-6)
            depth_loss = min(
                (1 - pearson_corrcoef( -zoe_depth, rendered_depth)),
                (1 - pearson_corrcoef(1 / (zoe_depth + 200.), rendered_depth))
                )
        # 选择 "dust3r" 类型, 使用 平均绝对误差 进行损失计算 使用 线性调度器 来调整深度损失权重
        elif mono_loss_type == "dust3r":

            # 创建掩码
            gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
            render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
            mask = torch.logical_and(gt_mask, render_mask)

            if mask.sum() < 10:
                depth_loss = 0.0
            else:
                disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                # 使用 平均绝对误差 进行损失计算
                depth_loss = torch.abs((disp_render - disp_mono)).mean()  
            #  使用 线性调度器 来调整深度损失权重
            depth_loss *= (opt.iterations - iteration) / opt.iterations # linear scheduler
        else:
            raise NotImplementedError

        # 总损失
        loss = loss + args.mono_depth_weight * depth_loss

    return loss, Ll1

def train_3dgs(args, ids):

    # 模型路径
    print("Optimizing " + args.model_path) 

    # Initialize system state (RNG)
    # 系统初始状态
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 加载数据集、管线、模型路径
    dataset = lp.extract(args)
    pipeline = pp.extract(args)
    model_path_root = args.model_path

    # 遍历稀疏视角图像编号进行 Leave-One-Out 训练
    for num_id, image_id in zip(range(args.sparse_view_num), ids): # num_id: leave one out id, image_id: the id of the image to be infered
        """ 
            num_id: 第几个留出（用于表示训练编号）
            image_id: 当前留出的图像 ID
        """
        # 留出的图像 模型保存路径
        args.model_path = os.path.join(model_path_root, f'leave_{image_id}')

        # 模型路径 同步给 数据集对象 （用于日志保存、图像输出）
        dataset.model_path = args.model_path

        # 要加载的起始权重
        args.start_checkpoint = os.path.join(args.model_path, 'chkpnt6000.pth') # load this ckpt
        # os.makedirs(args.model_path, exist_ok=True)
        leave_one_out_training(args,
                    dataset,
                    op.extract(args),
                    pipeline,
                    args.test_iterations,
                    args.save_iterations,
                    args.checkpoint_iterations,
                    args.start_checkpoint,
                    args.debug_from,
                    train_id = (num_id, image_id))


if __name__ == "__main__":
    # Set up command line argument parser
    # 命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")

    # 模型参数、优化器参数、训练管线参数
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # 添加自定义命令行参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_0000, 15_0000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    ### some exp args
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                    help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                    else use dense view. In sparse setting, sparse views will be used as training data, \
                    others will be used as testing data.")
    parser.add_argument("--use_mask", default=True, help="Use masked image, by default True")
    parser.add_argument('--use_dust3r', action='store_true', default=False,
                        help='use dust3r estimated poses')
    parser.add_argument('--dust3r_json', type=str, default=None)
    parser.add_argument("--init_pcd_name", default='origin', type=str, help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument('--mono_depth_weight', type=float, default=0.0005, help="The rate of monodepth loss")
    parser.add_argument('--mono_loss_type', type=str, default="mid")

    # 解析命令行参数
    args = parser.parse_args(sys.argv[1:])
    # 从命令行获取参数
    args.save_iterations.append(args.iterations)

    # 检查是否启用 稀疏视角训练 ，并加载相应图像编号
    assert args.sparse_view_num > 0, 'leave_one_out is for sparse view training'
    assert os.path.exists(os.path.join(args.source_path, f"sparse_{args.sparse_view_num}.txt")), f"sparse_{args.sparse_view_num}.txt not found!"

    # 加载图像编号列表（从 sparse_x.txt 文件中）
    ids = np.loadtxt(os.path.join(args.source_path, f"sparse_{args.sparse_view_num}.txt"), dtype=np.int32).tolist()

    # 启动训练，调用 leave-one-out 训练函数
    train_3dgs(args, ids)

    # All done
    print("\nAll training complete.")
