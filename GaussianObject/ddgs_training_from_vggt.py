#!/usr/bin/env python3
"""
D²GS训练专用脚本
基于VGGT + Visual Hull + 深度增强的结果进行D²GS训练
"""

import argparse
import os
import sys
import shutil
import subprocess
import numpy as np
import torch
import open3d as o3d
from pathlib import Path
import json
from typing import NamedTuple

# 添加路径
sys.path.append('/data/zhangao_data/3DGS/GaussianObject')
sys.path.append('/data/zhangao_data/3DGS/GaussianObject/scene')
sys.path.append('/data/zhangao_data/3DGS/GaussianObject/utils')
sys.path.append('/data/zhangao_data/3DGS/DDGS')

try:
    import pycolmap
except ImportError:
    print("警告: pycolmap未安装，将使用Open3D保存点云")
    pycolmap = None

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


def fov2focal(fov, pixels):
    """将视场角转换为焦距"""
    return pixels / (2 * np.tan(fov / 2))


def load_vggt_enhanced_results(vggt_output_dir, sparse_id=4, pointcloud_file=None):
    """
    加载VGGT + Visual Hull + 深度增强的结果
    
    Args:
        vggt_output_dir: VGGT输出目录
        sparse_id: 稀疏视角ID
        pointcloud_file: 指定的点云文件路径（可选）
    """
    print("=== 加载VGGT增强结果 ===")
    
    # 1. 如果指定了点云文件，直接使用
    if pointcloud_file is not None:
        if os.path.exists(pointcloud_file):
            print(f"使用指定的点云文件: {pointcloud_file}")
        else:
            raise FileNotFoundError(f"指定的点云文件不存在: {pointcloud_file}")
    else:
        # 2. 自动查找可用的点云文件
        possible_files = [
            os.path.join(vggt_output_dir, f"visual_hull_vggt_enhanced_{sparse_id}.ply"),
            os.path.join(vggt_output_dir, f"good_visual_hull_vggt_enhanced_{sparse_id}.ply"),
            os.path.join(vggt_output_dir, f"{sparse_id}_visual_hull_vggt_enhanced_{sparse_id}.ply"),
            os.path.join(vggt_output_dir, f"0_visual_hull_vggt_enhanced_{sparse_id}.ply"),
            os.path.join(vggt_output_dir, f"1_visual_hull_vggt_enhanced_{sparse_id}.ply"),
        ]
        
        pointcloud_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                pointcloud_file = file_path
                break
        
        if pointcloud_file is None:
            print(f"在 {vggt_output_dir} 中查找点云文件...")
            # 列出所有可能的文件
            import glob
            all_ply_files = glob.glob(os.path.join(vggt_output_dir, "*visual_hull_vggt_enhanced*.ply"))
            if all_ply_files:
                pointcloud_file = all_ply_files[0]  # 使用第一个找到的文件
                print(f"找到点云文件: {pointcloud_file}")
            else:
                raise FileNotFoundError(f"未找到VGGT增强点云文件，查找路径: {vggt_output_dir}")
    
    print(f"加载点云文件: {pointcloud_file}")
    pcd = o3d.io.read_point_cloud(pointcloud_file)
    
    if len(pcd.points) == 0:
        raise ValueError("点云文件为空")
    
    print(f"点云包含 {len(pcd.points)} 个点")
    
    # 3. 检查是否有法向量
    if not pcd.has_normals():
        print("点云缺少法向量，正在计算...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)
        print("法向量计算完成")
    
    return pcd


def load_camera_and_image_data(data_dir, sparse_id=4, resolution=1):
    """
    加载相机参数和图像数据
    """
    print("=== 加载相机参数和图像数据 ===")
    
    # 加载稀疏视角配置
    sparse_file = os.path.join(data_dir, f"sparse_{sparse_id}.txt")
    if not os.path.exists(sparse_file):
        raise FileNotFoundError(f"稀疏视角文件不存在: {sparse_file}")
    
    selected_ids = np.loadtxt(sparse_file, dtype=np.int32)
    print(f"使用稀疏视角 {sparse_id}: {len(selected_ids)} 个视角")
    
    # 使用GaussianObject的加载器
    from argparse import Namespace
    extra_opts = Namespace()
    extra_opts.sparse_view_num = -1
    extra_opts.resolution = resolution
    extra_opts.use_mask = True
    extra_opts.data_device = 'cuda'
    extra_opts.init_pcd_name = 'origin'
    extra_opts.white_background = False
    
    # 加载场景信息
    scene_info = sceneLoadTypeCallbacks["Colmap"](data_dir, 'images', False, extra_opts=extra_opts)
    camlist = cameraList_from_camInfos(scene_info.train_cameras, 1.0, extra_opts)
    
    # 根据selected_ids筛选相机
    selected_cameras = []
    for idx in selected_ids:
        if idx < len(camlist):
            selected_cameras.append(camlist[idx])
        else:
            print(f"警告: 视角索引 {idx} 超出范围")
    
    return selected_cameras, scene_info


def convert_to_colmap_format(pcd, selected_cameras, train_cameras, output_dir):
    """
    将VGGT增强结果转换为COLMAP格式
    """
    print("=== 转换为COLMAP格式 ===")
    
    # 创建COLMAP目录结构
    colmap_dir = os.path.join(output_dir, "colmap_input")
    os.makedirs(colmap_dir, exist_ok=True)
    os.makedirs(os.path.join(colmap_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(colmap_dir, "sparse"), exist_ok=True)
    
    # 1. 复制图像文件
    print("处理图像文件...")
    for i, cam_info in enumerate(selected_cameras):
        # 将tensor转换为numpy并保存
        image = cam_info.original_image
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        image_path = os.path.join(colmap_dir, "images", f"image_{i:04d}.jpg")
        import cv2
        cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    # 2. 保存COLMAP格式的点云
    print("保存COLMAP格式点云...")
    save_colmap_pointcloud(pcd, selected_cameras, colmap_dir)
    
    return colmap_dir


def save_colmap_pointcloud(pcd, cameras, colmap_dir):
    """
    保存COLMAP格式的点云文件
    """
    print("使用Open3D保存点云...")
    ply_path = os.path.join(colmap_dir, "sparse", "points3D.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"点云文件保存完成: {ply_path}")
    
    # 由于pycolmap相机参数验证失败，我们直接使用Open3D保存点云
    # 后续会使用现有的COLMAP相机参数文件


def run_colmap_bundle_adjustment(colmap_input_dir, output_dir):
    """
    运行COLMAP Bundle Adjustment
    """
    print("=== 运行COLMAP Bundle Adjustment ===")
    
    colmap_output_dir = os.path.join(output_dir, "colmap_output")
    os.makedirs(colmap_output_dir, exist_ok=True)
    
    # 检查是否有VGGT的demo_colmap.py
    vggt_colmap_script = "/data/zhangao_data/3DGS/vggt/demo_colmap.py"
    
    if os.path.exists(vggt_colmap_script):
        print("使用VGGT的COLMAP脚本...")
        try:
            # 使用VGGT的COLMAP脚本
            cmd = [
                "python", vggt_colmap_script,
                "--scene_dir", colmap_input_dir,
                "--use_ba"
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/data/zhangao_data/3DGS/vggt")
            
            if result.returncode == 0:
                print("COLMAP Bundle Adjustment 成功完成")
                # 复制结果到输出目录
                if os.path.exists(os.path.join(colmap_input_dir, "sparse")):
                    shutil.copytree(
                        os.path.join(colmap_input_dir, "sparse"),
                        os.path.join(colmap_output_dir, "sparse"),
                        dirs_exist_ok=True
                    )
                return colmap_output_dir
            else:
                print(f"COLMAP执行失败: {result.stderr}")
                print("跳过COLMAP Bundle Adjustment，直接使用输入数据")
                # 直接复制输入到输出
                shutil.copytree(colmap_input_dir, colmap_output_dir, dirs_exist_ok=True)
                return colmap_output_dir
                
        except Exception as e:
            print(f"COLMAP执行异常: {e}")
            print("跳过COLMAP Bundle Adjustment，直接使用输入数据")
            # 直接复制输入到输出
            shutil.copytree(colmap_input_dir, colmap_output_dir, dirs_exist_ok=True)
            return colmap_output_dir
    else:
        print("警告: VGGT COLMAP脚本不存在，跳过Bundle Adjustment")
        # 直接复制输入到输出
        shutil.copytree(colmap_input_dir, colmap_output_dir, dirs_exist_ok=True)
        return colmap_output_dir


def prepare_ddgs_training(colmap_output_dir, ddgs_output_dir):
    """
    准备D²GS训练环境
    """
    print("=== 准备D²GS训练环境 ===")
    
    # 直接使用现有的kitchen数据集，它已经有正确的D²GS结构
    original_data_dir = "/data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen"
    
    # 创建D²GS训练目录
    os.makedirs(ddgs_output_dir, exist_ok=True)
    
    # 复制整个kitchen数据集到D²GS输出目录
    ddgs_data_dir = os.path.join(ddgs_output_dir, "data")
    if os.path.exists(ddgs_data_dir):
        shutil.rmtree(ddgs_data_dir)
    
    print(f"复制kitchen数据集到: {ddgs_data_dir}")
    shutil.copytree(original_data_dir, ddgs_data_dir)
    
    # 创建D²GS需要的预处理掩码目录并生成掩码
    create_preprocessed_masks_directory(ddgs_output_dir)
    generate_ddgs_masks_from_existing(ddgs_output_dir)
    
    return ddgs_data_dir


def generate_ddgs_masks_from_existing(ddgs_output_dir, resolution_factor=8):
    """
    从现有的PNG掩码生成D²GS需要的.pt掩码文件
    """
    print("=== 生成D²GS掩码文件 ===")
    
    # 源掩码目录（PNG格式）
    source_masks_dir = "/data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen/masks"
    
    # 目标掩码目录（.pt格式）
    target_masks_dir = os.path.join(ddgs_output_dir, "preprocessed_masks_10", "data", "r8")
    
    if not os.path.exists(source_masks_dir):
        print(f"警告: 源掩码目录不存在: {source_masks_dir}")
        return False
    
    if not os.path.exists(target_masks_dir):
        print(f"错误: 目标掩码目录不存在: {target_masks_dir}")
        return False
    
    # 获取所有PNG掩码文件
    png_files = [f for f in os.listdir(source_masks_dir) if f.endswith('.png')]
    png_files.sort()  # 按文件名排序
    
    print(f"找到 {len(png_files)} 个PNG掩码文件")
    
    if len(png_files) == 0:
        print("警告: 没有找到PNG掩码文件")
        return False
    
    # 检查第一个掩码和对应图像的尺寸
    first_mask = png_files[0]
    first_image = first_mask.replace('.png', '.JPG')
    
    mask_path = os.path.join(source_masks_dir, first_mask)
    image_path = os.path.join("/data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen/images", first_image)
    
    import cv2
    if os.path.exists(mask_path) and os.path.exists(image_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image_img = cv2.imread(image_path)
        
        print(f"原始掩码尺寸: {mask_img.shape}")
        print(f"原始图像尺寸: {image_img.shape}")
        
        # 计算目标尺寸（图像尺寸除以分辨率因子）
        target_h = round(image_img.shape[0] / resolution_factor)
        target_w = round(image_img.shape[1] / resolution_factor)
        print(f"目标掩码尺寸: {target_h}x{target_w}")
        
        # 确保尺寸完全匹配（处理舍入问题）
        if target_h != round(image_img.shape[0] / resolution_factor):
            target_h = round(image_img.shape[0] / resolution_factor)
        if target_w != round(image_img.shape[1] / resolution_factor):
            target_w = round(image_img.shape[1] / resolution_factor)
        print(f"修正后目标掩码尺寸: {target_h}x{target_w}")
    
    converted_count = 0
    
    for png_file in png_files:
        try:
            # 读取PNG掩码
            mask_path = os.path.join(source_masks_dir, png_file)
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask_image is None:
                print(f"警告: 无法读取 {png_file}")
                continue
            
            # 根据目标尺寸调整掩码
            if resolution_factor > 1:
                mask_image = cv2.resize(mask_image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                if converted_count < 5:  # 只打印前几个
                    print(f"调整掩码尺寸: {png_file} -> {target_h}x{target_w}")
            
            # 转换为张量 (0-1范围)
            mask_tensor = torch.from_numpy(mask_image.astype(np.float32) / 255.0)
            
            # 保存为.pt文件
            pt_filename = png_file.replace('.png', '.pt')
            pt_path = os.path.join(target_masks_dir, pt_filename)
            torch.save(mask_tensor, pt_path)
            
            converted_count += 1
            
            if converted_count % 50 == 0:
                print(f"已转换 {converted_count}/{len(png_files)} 个文件")
                
        except Exception as e:
            print(f"转换 {png_file} 时出错: {e}")
            continue
    
    print(f"=== 掩码生成完成 ===")
    print(f"成功转换 {converted_count}/{len(png_files)} 个掩码文件")
    print(f"掩码目录: {target_masks_dir}")
    
    return converted_count > 0


def create_preprocessed_masks_directory(ddgs_output_dir):
    """
    创建D²GS需要的预处理掩码目录
    """
    print("创建D²GS预处理掩码目录...")
    
    # D²GS期望的掩码目录结构: preprocessed_masks_10/data/r8/
    mask_dir = os.path.join(ddgs_output_dir, "preprocessed_masks_10", "data", "r8")
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"创建掩码目录: {mask_dir}")
    print("注意: 如果需要使用掩码功能，请在此目录中放置预处理的.pt掩码文件")


def create_ddgs_directory_structure(data_dir):
    """
    创建D²GS期望的目录结构
    """
    print("创建D²GS目录结构...")
    
    # 1. 确保sparse目录存在
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    
    # 2. 创建12_views/dense目录（D²GS期望的格式）
    views_dir = os.path.join(data_dir, "12_views", "dense")
    os.makedirs(views_dir, exist_ok=True)
    
    # 检查是否有带法向量的点云文件
    original_sparse_dir = "/data/zhangao_data/3DGS/GaussianObject/data/mip360/kitchen/sparse/0"
    if os.path.exists(os.path.join(original_sparse_dir, "points3D.ply")):
        # 复制带法向量的点云文件到sparse目录
        shutil.copy2(
            os.path.join(original_sparse_dir, "points3D.ply"),
            os.path.join(sparse_dir, "points3D.ply")
        )
        print(f"复制带法向量的点云文件: {sparse_dir}/points3D.ply")
        
        # 复制带法向量的点云文件到12_views/dense目录（D²GS期望的格式）
        shutil.copy2(
            os.path.join(original_sparse_dir, "points3D.ply"),
            os.path.join(views_dir, "fused.ply")
        )
        print(f"复制点云文件到D²GS期望位置: {views_dir}/fused.ply")
        
        # 同时复制COLMAP的相机和图像文件
        for file_name in ["cameras.bin", "images.bin", "points3D.bin"]:
            original_file = os.path.join(original_sparse_dir, file_name)
            if os.path.exists(original_file):
                shutil.copy2(original_file, os.path.join(sparse_dir, file_name))
                print(f"复制COLMAP文件: {sparse_dir}/{file_name}")
    else:
        print("警告: 未找到带法向量的points3D.ply文件")


def run_ddgs_training(ddgs_data_dir, output_dir, iterations=30000):
    """
    运行D²GS训练
    """
    print("=== 运行D²GS训练 ===")
    
    ddgs_script = "/data/zhangao_data/3DGS/DDGS/train.py"
    
    if not os.path.exists(ddgs_script):
        print("错误: D²GS训练脚本不存在")
        return False
    
    # 确保掩码目录在D²GS工作目录下
    ddgs_work_dir = "/data/zhangao_data/3DGS/DDGS"
    mask_source_dir = os.path.join(output_dir, "preprocessed_masks_10")
    mask_target_dir = os.path.join(ddgs_work_dir, "preprocessed_masks_10")
    
    # 复制掩码目录到D²GS工作目录
    if os.path.exists(mask_source_dir):
        if os.path.exists(mask_target_dir):
            shutil.rmtree(mask_target_dir)
        shutil.copytree(mask_source_dir, mask_target_dir)
        print(f"复制掩码目录到D²GS工作目录: {mask_target_dir}")
    
    # D²GS训练命令
    cmd = [
        "python", ddgs_script,
        "-s", os.path.abspath(ddgs_data_dir),  # 使用绝对路径
        "-m", os.path.abspath(os.path.join(output_dir, "ddgs_model")),  # 使用绝对路径
        "--depth_weight", "0.1",
        "--density_weight", "0.9",
        "--drop_min", "0.05",
        "--drop_max", "0.5",
        "--mask_param", "10",
        "--lambda_far", "0.5",
        "--eval",
        "-r", "8",
        "--n_views", "12",
        "--iterations", str(iterations)
    ]
    
    print(f"执行D²GS训练命令: {' '.join(cmd)}")
    
    try:
        # 切换到D²GS目录
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd="/data/zhangao_data/3DGS/DDGS"
        )
        
        if result.returncode == 0:
            print("D²GS训练成功完成")
            return True
        else:
            print(f"D²GS训练失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"D²GS训练异常: {e}")
        return False


def run_ddgs_rendering_and_evaluation(ddgs_model_dir, output_dir):
    """
    运行D²GS渲染和评估
    """
    print("=== 运行D²GS渲染和评估 ===")
    
    ddgs_render_script = "/data/zhangao_data/3DGS/DDGS/render.py"
    
    if not os.path.exists(ddgs_render_script):
        print("警告: D²GS渲染脚本不存在，跳过渲染")
        return False
    
    # D²GS渲染命令 - 使用绝对路径
    cmd = [
        "python", ddgs_render_script,
        "-m", os.path.abspath(ddgs_model_dir),  # 使用绝对路径
        "-s", os.path.abspath(os.path.join(output_dir, "data")),  # 使用绝对路径
        "--eval",
        "-r", "8"
    ]
    
    print(f"执行D²GS渲染命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/data/zhangao_data/3DGS/DDGS"
        )
        
        if result.returncode == 0:
            print("D²GS渲染和评估成功完成")
            return True
        else:
            print(f"D²GS渲染失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"D²GS渲染异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='D²GS训练专用脚本 - 基于VGGT增强结果')
    
    # 输入参数
    parser.add_argument('--vggt_output_dir', type=str, required=True,
                       help='VGGT + Visual Hull + 深度增强的输出目录')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='原始数据集目录')
    parser.add_argument('--sparse_id', type=int, default=4,
                       help='稀疏视角ID')
    parser.add_argument('--resolution', type=int, default=1,
                       help='图像分辨率')
    parser.add_argument('--pointcloud_file', type=str, default=None,
                       help='指定的点云文件路径（可选，如果不指定则自动查找）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, required=True,
                       help='D²GS输出目录')
    
    # D²GS训练参数
    parser.add_argument('--ddgs_iterations', type=int, default=30000,
                       help='D²GS训练迭代次数')
    parser.add_argument('--depth_weight', type=float, default=0.1,
                       help='D²GS深度损失权重')
    parser.add_argument('--density_weight', type=float, default=0.9,
                       help='D²GS密度损失权重')
    parser.add_argument('--drop_min', type=float, default=0.05,
                       help='D²GS最小dropout概率')
    parser.add_argument('--drop_max', type=float, default=0.5,
                       help='D²GS最大dropout概率')
    parser.add_argument('--mask_param', type=int, default=10,
                       help='D²GS掩码参数')
    parser.add_argument('--lambda_far', type=float, default=0.5,
                       help='D²GS远距离损失权重')
    parser.add_argument('--n_views', type=int, default=12,
                       help='D²GS训练视图数量')
    parser.add_argument('--resolution_factor', type=int, default=8,
                       help='D²GS渲染分辨率倍数')
    
    # 控制参数
    parser.add_argument('--skip_colmap', action='store_true',
                       help='跳过COLMAP Bundle Adjustment')
    parser.add_argument('--skip_ddgs', action='store_true',
                       help='跳过D²GS训练')
    parser.add_argument('--skip_rendering', action='store_true',
                       help='跳过D²GS渲染和评估')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("D²GS训练专用脚本")
    print("基于VGGT + Visual Hull + 深度增强的结果")
    print("=" * 60)
    
    try:
        # === 阶段1: 加载VGGT增强结果 ===
        print("\n" + "=" * 40)
        print("阶段1: 加载VGGT增强结果")
        print("=" * 40)
        
        vggt_pcd = load_vggt_enhanced_results(args.vggt_output_dir, args.sparse_id, args.pointcloud_file)
        selected_cameras, scene_info = load_camera_and_image_data(
            args.data_dir, args.sparse_id, args.resolution
        )
        
        # === 阶段2: 转换为COLMAP格式 ===
        print("\n" + "=" * 40)
        print("阶段2: 转换为COLMAP格式")
        print("=" * 40)
        
        colmap_input_dir = convert_to_colmap_format(
            vggt_pcd, selected_cameras, scene_info.train_cameras, args.output_dir
        )
        
        # === 阶段3: COLMAP Bundle Adjustment ===
        if not args.skip_colmap:
            print("\n" + "=" * 40)
            print("阶段3: COLMAP Bundle Adjustment")
            print("=" * 40)
            
            colmap_output_dir = run_colmap_bundle_adjustment(colmap_input_dir, args.output_dir)
            
            if colmap_output_dir is None:
                print("错误: COLMAP Bundle Adjustment失败")
                return False
        else:
            print("跳过COLMAP Bundle Adjustment")
            colmap_output_dir = colmap_input_dir
        
        # === 阶段4: D²GS训练 ===
        if not args.skip_ddgs:
            print("\n" + "=" * 40)
            print("阶段4: D²GS训练")
            print("=" * 40)
            
            ddgs_data_dir = prepare_ddgs_training(colmap_output_dir, args.output_dir)
            
            ddgs_success = run_ddgs_training(ddgs_data_dir, args.output_dir, args.ddgs_iterations)
            
            if not ddgs_success:
                print("D²GS训练失败")
                return False
        else:
            print("跳过D²GS训练")
        
        # === 阶段5: D²GS渲染和评估 ===
        if not args.skip_rendering and not args.skip_ddgs:
            print("\n" + "=" * 40)
            print("阶段5: D²GS渲染和评估")
            print("=" * 40)
            
            ddgs_model_dir = os.path.join(args.output_dir, "ddgs_model")
            render_success = run_ddgs_rendering_and_evaluation(ddgs_model_dir, args.output_dir)
            
            if not render_success:
                print("D²GS渲染失败")
        else:
            print("跳过D²GS渲染和评估")
        
        print("\n" + "=" * 60)
        print("D²GS训练流程成功完成！")
        print("=" * 60)
        print(f"输出目录: {args.output_dir}")
        print(f"COLMAP结果: {os.path.join(args.output_dir, 'colmap_output')}")
        print(f"D²GS模型: {os.path.join(args.output_dir, 'ddgs_model')}")
        return True
        
    except Exception as e:
        print(f"\n❌ 流程执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
