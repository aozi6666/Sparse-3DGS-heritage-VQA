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

    # 初始化
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False, extra_opts=args) # make sure no shuffle
    gaussians.training_setup(opt)

    # 检查点路径
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 背景颜色设置
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 训练迭代所需的事件
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 初始化变量 num_id是“留一法”的索引
    num_id, image_id = train_id   
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    # 训练进度条初始化
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # 训练循环
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record() # type: ignore

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每10000步增加一次球面谐波（SH）的阶数
        if iteration % 10000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera 选择一个随机视角 “留一法”训练
        if not viewpoint_stack:
            if iteration <= opt.densify_until_iter:
                viewpoint_stack = scene.getTrainCameras().copy()[:num_id] + scene.getTrainCameras().copy()[num_id+1:] # leave one out
            else:
                viewpoint_stack = scene.getTrainCameras().copy() # after the densify_until_iter, use all the cameras
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render 渲染图像
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss 计算损失
        loss, Ll1 = cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, mono_loss_type=args.mono_loss_type, iteration=iteration)

        loss.backward()

        # 记录并更新进度
        iter_end.record()  # type: ignore

        with torch.no_grad():
            # Progress bar
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

            # Densification 稠密化过程：更新模型的密度并进行修剪
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                # if iteration % (opt.opacity_reset_interval // 2) == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                if iteration % opt.remove_outliers_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.remove_outliers(opt, iteration, linear=True)

            # Optimizer step 优化步骤，更新高斯模型的参数
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # cache the rendered images in the rendering progress.
            # 渲染测试图像并保存
            if iteration > opt.densify_until_iter and not viewpoint_stack: 
                infer_cam = scene.getTrainCameras().copy()[num_id]
                bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                rendering = render(infer_cam, gaussians, pipe, background)["render"]
                gt = infer_cam.original_image[0:3, :, :]
                render_path = os.path.join(args.model_path, 'left_image')
                os.makedirs(render_path, exist_ok=True)
                torchvision.utils.save_image(rendering, os.path.join(render_path, f'sample_{iteration}.png'))
                if not os.path.exists(os.path.join(args.model_path, 'gt.png')):
                    torchvision.utils.save_image(gt, os.path.join(args.model_path, 'gt.png'))

    ### in the end, we need store the gaussians.cache for latter use 
    #  最终保存模型
    torch.save(gaussians.cache, os.path.join(args.model_path, 'gaussians_cache.pth'))

# 创建日志文件 初始化TensorBoard记录器 
def prepare_output_and_logger(args):  

    # 判断 args.model_path 是否存在  
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
            args.model_path = os.path.join("./output/", unique_str)
        else:
            unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder 设置输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    # 保存配置参数
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    # 初始化 TensorBoard
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# 训练记录
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    
    # 记录训练损失到TensorBoard
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    # 在指定的测试迭代时进行测试
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        # 设置验证集配置
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # 遍历验证集配置
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                # 遍历相机视角进行评估渲染
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # 将前5张图像写入 TensorBoard（渲染图和GT图）
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    # 累加当前视角的误差（L1 和 PSNR）
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                # 求平均损失并输出日志
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
               # 将评估指标写入 TensorBoard
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
       
        # 记录当前场景高斯球的不透明度分布和数量
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid", iteration=0):
    """
    Calculate the loss of the image, contains l1 loss and ssim loss.

    l1 loss: Ll1 = l1_loss(image, gt_image)
    ssim loss: Lssim = 1 - ssim(image, gt_image)

    Optional: [silhouette loss, monodepth loss]
    ilhouette_loss_type: 轮廓损失类型（默认为 "bce"，即二元交叉熵）
    mono_loss_type: 单目深度损失类型（默认为 "mid"）
    """

    # 获取 ground-truth 图像
    gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()

    # 处理随机背景
    if opt.random_background:
        gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()
    
    # 计算 L1 损失  SSIM 损失
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    
    # 计算轮廓损失
    if hasattr(args, "use_mask") and args.use_mask:
        if silhouette_loss_type == "bce":
            silhouette_loss = F.binary_cross_entropy(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        elif silhouette_loss_type == "mse":
            silhouette_loss = F.mse_loss(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        else:
            raise NotImplementedError
        loss = loss + opt.lambda_silhouette * silhouette_loss
    
    # 计算单目深度损失
    if hasattr(viewpoint_cam, "mono_depth") and  viewpoint_cam.mono_depth is not None:
        if mono_loss_type == "mid":

            # we apply masked monocular loss
            # 创建像素掩码 mask
            gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
            render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
            mask = torch.logical_and(gt_mask, render_mask)
            if mask.sum() < 10:
                depth_loss = 0.0
            else:
                disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]

        #  "pearson" 类型的单目深度损失,
        # 计算 zoe_depth（来自相机的单目深度）和 rendered_depth（渲染深度）之间的 Pearson 相关系数        
        elif mono_loss_type == "pearson":
            zoe_depth = viewpoint_cam.mono_depth[viewpoint_cam.mask > 0.5].clamp(1e-6)
            rendered_depth = render_pkg["rendered_depth"][viewpoint_cam.mask > 0.5].clamp(1e-6)
            depth_loss = min(
                (1 - pearson_corrcoef( -zoe_depth, rendered_depth)),
                (1 - pearson_corrcoef(1 / (zoe_depth + 200.), rendered_depth))
                )
            
        # 选择 "dust3r" 类型的单目深度损失，计算深度图的绝对误差
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

        # 深度损失加到总损失中
        loss = loss + args.mono_depth_weight * depth_loss

    return loss, Ll1

def train_3dgs(args, ids):

    # 模型路径
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG) 
    # 初始化系统状态（随机数生成器）
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 加载数据集和管道
    dataset = lp.extract(args)
    pipeline = pp.extract(args)
    model_path_root = args.model_path  # 模型路径根目录
    
    # 遍历 ids 列表进行留一法训练
    for num_id, image_id in zip(range(args.sparse_view_num), ids): 
        """
        # num_id: leave one out id, image_id: the id of the image to be infered
        (num_id, image_id) 对。num_id 是留一法中的索引（即当前迭代要“留”的图像 ID）
                              image_id 是当前要推理的图像 ID
        """

        # 设置训练模型路径
        args.model_path = os.path.join(model_path_root, f'leave_{image_id}')

        # 更新数据集模型路径
        dataset.model_path = args.model_path

        # 创建模型输出目录
        os.makedirs(args.model_path, exist_ok=True)

        # 调用 leave_one_out_training 函数开始训练
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
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")

    # 加载模型、优化和管道参数
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # 命令行参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_0000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_0000, 15_0000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[6000])
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
    # 将训练迭代次数加入保存迭代次数
    args.save_iterations.append(args.iterations)

    # 确保 sparse_view_num 大于 0
    assert args.sparse_view_num > 0, 'leave_one_out is for sparse view training'

    # 验证稀疏视图文件是否存在
    assert os.path.exists(os.path.join(args.source_path, f"sparse_{args.sparse_view_num}.txt")), f"sparse_{args.sparse_view_num}.txt not found!"

    # 加载稀疏视图的图像 ID
    ids = np.loadtxt(os.path.join(args.source_path, f"sparse_{args.sparse_view_num}.txt"), dtype=np.int32).tolist()

    # 调用 train_3dgs 函数开始训练
    train_3dgs(args, ids)

    # All done
    print("\nAll training complete.")
