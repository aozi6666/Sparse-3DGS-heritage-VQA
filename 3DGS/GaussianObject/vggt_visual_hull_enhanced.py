#!/usr/bin/env python3
"""
VGGT深度增强的视觉外壳生成脚本
分阶段设计：基础视觉体 + VGGT深度增强 + 点云合并
"""

import argparse
import math
import os
import sys
import torch
import numpy as np
import open3d as o3d
from tqdm import trange
import cv2
from torch.nn import functional as F
from typing import NamedTuple
from torchvision import transforms
from argparse import Namespace

# Paths relative to this file: .../3DGS/GaussianObject/vggt_visual_hull_enhanced.py
_GO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_3DGS = os.path.dirname(_GO_DIR)  # .../3DGS
_VGGT_DIR = os.path.join(_REPO_3DGS, 'vggt')
sys.path.append(_GO_DIR)
sys.path.append(os.path.join(_GO_DIR, 'scene'))
sys.path.append(os.path.join(_GO_DIR, 'utils'))
sys.path.append(_VGGT_DIR)

try:
    from vggt.models.vggt import VGGT
except ImportError as e:
    print(f"导入VGGT模块失败: {e}")
    sys.exit(1)

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


def fov2focal(fov, pixels):
    """将视场角转换为焦距"""
    return pixels / (2 * math.tan(fov / 2))


def points2homopoints(points):
    """将3D点转换为齐次坐标"""
    assert points.shape[-1] == 3
    bottom = torch.ones_like(points[...,0:1])
    return torch.cat([points, bottom], dim=-1)


def batch_projection(Ks, Ts, points):
    """批量投影函数"""
    pre_fix = points.shape[:-1]
    points = points.reshape(-1, 3)
    
    Ts = torch.stack(Ts, dim=0)
    Ks = torch.stack(Ks, dim=0).to(Ts.device)
    camera_num = Ks.shape[0]
    homopts = points2homopoints(points)
    
    # world to camera
    homopts_cam = torch.bmm(homopts.unsqueeze(0).repeat_interleave(Ts.shape[0], dim=0), Ts.transpose(1,2))
    # camera to image space
    homopts_img = torch.bmm(homopts_cam[...,:3], Ks.transpose(1,2))
    # normalize
    homopts_img = homopts_img / (homopts_img[...,2:] + 1e-6)
    # reshape back
    homopts_img = homopts_img.reshape(camera_num, *pre_fix, 3)
    homopts_cam = homopts_cam.reshape(camera_num, *pre_fix, 4)
    return homopts_img[...,0:2], homopts_cam[...,2]


def simple_resize_image(img, size):
    """图像缩放"""
    return transforms.Resize(size, antialias=True)(img)


def load_existing_depth_maps(data_dir, selected_ids, target_size):
    """加载现有深度图"""
    print("加载现有深度图...")
    depth_maps = []
    
    zoe_depth_dir = os.path.join(data_dir, "zoe_depth")
    if not os.path.exists(zoe_depth_dir):
        print(f"警告: zoe_depth目录不存在: {zoe_depth_dir}")
        return [torch.zeros(1, target_size[0], target_size[1]) for _ in selected_ids]
    
    depth_files = [f for f in os.listdir(zoe_depth_dir) if f.endswith('.png')]
    depth_files.sort()
    
    for idx in selected_ids:
        if idx < len(depth_files):
            depth_filename = depth_files[idx]
            depth_path = os.path.join(zoe_depth_dir, depth_filename)
            
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_map is not None:
                depth_map = cv2.resize(depth_map, (target_size[1], target_size[0]))
                depth_map = depth_map.astype(np.float32)
                if depth_map.max() > 1000:
                    depth_map = depth_map / 1000.0
                depth_maps.append(torch.from_numpy(depth_map).float().unsqueeze(0))
            else:
                depth_maps.append(torch.zeros(1, target_size[0], target_size[1]))
        else:
            depth_maps.append(torch.zeros(1, target_size[0], target_size[1]))
    
    return depth_maps


def evaluate_depth_quality_with_vggt(images, depth_maps, model, device):
    """使用VGGT评估深度图质量；同时返回逐视角深度（CPU），供后续增强复用，避免二次整批推理 OOM。"""
    print("使用VGGT评估深度图质量...")
    
    # 调整图像尺寸以适配VGGT
    images_resized, _, _ = adjust_image_size_for_vggt(images, [], [])
    target_size = (images_resized[0].shape[1], images_resized[0].shape[2])
    
    # 调整深度图尺寸
    depth_maps_resized = []
    for depth_map in depth_maps:
        depth_np = depth_map.squeeze(0).cpu().numpy()
        depth_resized = cv2.resize(depth_np, (target_size[1], target_size[0]))
        depth_maps_resized.append(torch.from_numpy(depth_resized).float().unsqueeze(0))
    
    quality_scores = []
    vggt_depth_list = []
    for i, (image, depth_map) in enumerate(zip(images_resized, depth_maps_resized)):
        image_tensor = image.unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            results = model(image_tensor)
        
        vggt_depth = results['depth']
        if len(vggt_depth.shape) == 5:
            vggt_depth = vggt_depth.squeeze(0).squeeze(-1)
        elif len(vggt_depth.shape) == 4:
            vggt_depth = vggt_depth.squeeze(1)
        vggt_depth = vggt_depth.squeeze(0)  # [H, W]
        
        existing_depth = depth_map.squeeze(0).to(device)
        depth_diff = torch.abs(existing_depth - vggt_depth)
        quality_score = 1.0 / (1.0 + depth_diff.mean().item())
        
        quality_scores.append(quality_score)
        vggt_depth_list.append(vggt_depth.detach().cpu())
        print(f"视角 {i} 深度图质量分数: {quality_score:.3f}")

        del results, image_tensor, existing_depth, depth_diff, vggt_depth
        torch.cuda.empty_cache()
    
    vggt_depths = torch.stack(vggt_depth_list, dim=0)  # [N, H, W] on CPU
    return quality_scores, vggt_depths


def adjust_image_size_for_vggt(images, masks, Ks, patch_size=14):
    """调整图像尺寸以适配VGGT模型"""
    H, W = images[0].shape[1], images[0].shape[2]
    new_H = ((H + patch_size - 1) // patch_size) * patch_size
    new_W = ((W + patch_size - 1) // patch_size) * patch_size
    
    scale_x = new_W / W
    scale_y = new_H / H
    
    images_resized = []
    for img in images:
        img_resized = F.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False)
        images_resized.append(img_resized.squeeze(0))
    
    masks_resized = []
    for mask in masks:
        mask_resized = F.interpolate(mask.unsqueeze(0), size=(new_H, new_W), mode='nearest')
        mask_resized = mask_resized * 255.0
        masks_resized.append(mask_resized.squeeze(0))
    
    Ks_adjusted = []
    for K in Ks:
        K_new = K.copy()
        K_new[0, 0] *= scale_x
        K_new[1, 1] *= scale_y
        K_new[0, 2] *= scale_x
        K_new[1, 2] *= scale_y
        Ks_adjusted.append(K_new)
    
    return images_resized, masks_resized, Ks_adjusted


def precompute_vggt_depths(images, model, device):
    """逐张预计算 VGGT 深度图（避免多视角整批推理 OOM）。返回 CPU tensor [N,H,W]。"""
    print("预计算VGGT深度图（逐张推理）...")
    torch.cuda.empty_cache()

    depth_list = []
    for i, img in enumerate(images):
        H, W = img.shape[1], img.shape[2]
        patch_size = 14
        new_H = ((H + patch_size - 1) // patch_size) * patch_size
        new_W = ((W + patch_size - 1) // patch_size) * patch_size
        img_resized = F.interpolate(
            img.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False
        ).squeeze(0)

        image_tensor = img_resized.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,C,H,W]
        with torch.no_grad():
            results = model(image_tensor)

        vggt_depth = results['depth']
        if len(vggt_depth.shape) == 5:
            vggt_depth = vggt_depth.squeeze(0).squeeze(-1)
        elif len(vggt_depth.shape) == 4:
            vggt_depth = vggt_depth.squeeze(1)
        vggt_depth = vggt_depth.squeeze(0)  # [H, W]

        depth_list.append(vggt_depth.detach().cpu())
        print(f"  视角 {i+1}/{len(images)} 深度预计算完成")
        del results, image_tensor, img_resized, vggt_depth
        torch.cuda.empty_cache()

    vggt_depths = torch.stack(depth_list, dim=0)
    print(f"VGGT深度图预计算完成，形状: {vggt_depths.shape}")
    return vggt_depths


# ==================== 阶段1: 基础视觉体生成 ====================
def get_visual_hull_original(N, bbox, scene_info, cam_center, device):
    """阶段1: 基础视觉体生成（保持原始多视角一致性逻辑）"""
    print("=== 阶段1: 生成基础视觉体（保持原始多视角一致性） ===")
    
    pcs = []
    color = []
    Ks = scene_info.Ks
    Ts = scene_info.Ts
    images = scene_info.images
    masks = scene_info.masks

    [xs, ys, zs], [xe, ye, ze] = bbox[0], bbox[1]
    print(f"边界框: X=[{xs:.2f}, {xe:.2f}], Y=[{ys:.2f}, {ye:.2f}], Z=[{zs:.2f}, {ze:.2f}]")

    # 图像和掩码进行统一大小
    new_images = []
    new_masks = []
    img_size = images[0].shape[1:]
    for image, mask in zip(images, masks):
        new_images.append(simple_resize_image(image, img_size))
        new_masks.append(simple_resize_image(mask, img_size))

    images = torch.stack(new_images)
    masks = torch.stack(new_masks)

    # 生成2D网格
    for h_id in trange(N):
        i, j = torch.meshgrid(torch.linspace(xs, xe, N).to(device),
                              torch.linspace(ys, ye, N).to(device), indexing='ij')
        i, j = i.t(), j.t()
        pts = torch.stack([i, j, torch.ones_like(i).to(device)], -1)
        pts[...,2] = h_id / N * (ze - zs) + zs

        # 平移坐标到相机中心
        pts[...,0] += cam_center[0]
        pts[...,1] += cam_center[1]
        pts[...,2] += cam_center[2]

        # 投影到图像平面
        uv, z = batch_projection(Ks, Ts, pts)

        # 判断有效点
        valid_z_mask = z > 0
        valid_x_y_mask = (uv[...,0] > 0) & (uv[...,0] < images.shape[3]) & (uv[...,1] > 0) & (uv[...,1] < images.shape[2])
        valid_pt_mask = valid_z_mask & valid_x_y_mask

        # 归一化图像坐标
        uv[...,0] = uv[...,0] / images.shape[3] * 2 - 1
        uv[...,1] = uv[...,1] / images.shape[2] * 2 - 1

        # 从图像中采样颜色
        result = F.grid_sample(images.float(), uv, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1)
        result_mask = F.grid_sample(masks.float(), uv, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1)

        # 生成视觉体 - 使用原始的多视角一致性要求
        valid_pt_mask = result_mask.squeeze() > 0 & valid_pt_mask
        pcs.append(valid_pt_mask.float().sum(0) >= (images.shape[0] - 1))  # 需要N-1个视角都看到
        color.append(result.mean(0))

    # 转换为点云
    pcs = torch.stack(pcs, -1)
    color = torch.stack(color, -1)
    
    r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    idx = torch.where(pcs > 0)

    print(f"基础视觉体生成完成，包含 {len(idx[0])} 个点")

    if len(idx[0]) == 0:
        print("警告: 没有点满足多视角可见性要求！")
        return None, None

    color = torch.stack((r[idx] * 255, g[idx] * 255, b[idx] * 255), -1)
    idx = torch.stack([idx[1], idx[0], idx[2]], -1)
    
    # 将idx转换为实际坐标
    idx = idx.float() / N
    idx[...,0] = idx[...,0] * (xe - xs) + xs + cam_center[0]
    idx[...,1] = idx[...,1] * (ye - ys) + ys + cam_center[1]
    idx[...,2] = idx[...,2] * (ze - zs) + zs + cam_center[2]

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(idx.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy() / 255)

    return pcd, pcd.get_axis_aligned_bounding_box()


def backproject_point(u, v, depth, K, T):
    """将2D点反投影到3D空间"""
    try:
        point_2d = torch.tensor([u, v, 1.0]).float().to(K.device)
        K_inv = torch.inverse(K)
        point_cam = depth * (K_inv @ point_2d)
        T_inv = torch.inverse(T)
        point_world = T_inv @ torch.cat([point_cam, torch.ones(1).to(K.device)])
        return point_world[:3].cpu().numpy()
    except Exception as e:
        return None


def check_and_generate_vggt_enhancement(point, color, scene_info, depth_maps, vggt_depths, quality_scores, model, device,
                                      vggt_quality_threshold, vggt_enhancement_factor):
    """检查点是否应该被VGGT增强并生成增强点（使用预计算的深度图）"""
    Ks = scene_info.Ks
    Ts = scene_info.Ts
    images = scene_info.images
    
    # 转换numpy数组为PyTorch张量并确保在正确设备上
    Ks = [torch.from_numpy(K).float().to(device) if isinstance(K, np.ndarray) else K.to(device) for K in Ks]
    Ts = [torch.from_numpy(T).float().to(device) if isinstance(T, np.ndarray) else T.to(device) for T in Ts]
    
    # 将3D点转换为齐次坐标并确保在正确设备上
    point_tensor = torch.from_numpy(point).float().to(device)
    point_homo = torch.cat([point_tensor, torch.ones(1).to(device)])
    
    # 投影到各个视角
    projected_points = []
    depth_consistencies = []
    
    for view_idx in range(len(images)):
        if view_idx >= len(depth_maps) or quality_scores[view_idx] < vggt_quality_threshold:
            continue
        
        # 投影到当前视角
        T = Ts[view_idx]
        K = Ks[view_idx]
        
        # World to camera
        point_cam = T @ point_homo
        if point_cam[2] <= 0:
            continue
        
        # Camera to image - 确保point_cam[:3]在正确设备上
        point_cam_3d = point_cam[:3].to(device)
        point_img = K @ point_cam_3d
        u = point_img[0] / point_img[2]
        v = point_img[1] / point_img[2]
        
        # 检查图像维度并正确访问
        image_shape = images[view_idx].shape
        if len(image_shape) == 4:  # [C, H, W, ?] 或 [N, C, H, W]
            img_width = image_shape[3]
            img_height = image_shape[2]
        elif len(image_shape) == 3:  # [C, H, W]
            img_width = image_shape[2]
            img_height = image_shape[1]
        else:
            print(f"警告: 图像 {view_idx} 维度不正确: {image_shape}")
            continue
        
        # 检查是否在图像范围内
        if u < 0 or u >= img_width or v < 0 or v >= img_height:
            continue
        
        # 采样原始深度图
        depth_map = depth_maps[view_idx].squeeze(0).to(device)
        u_int = torch.clamp(torch.tensor(u).long(), 0, depth_map.shape[1] - 1)
        v_int = torch.clamp(torch.tensor(v).long(), 0, depth_map.shape[0] - 1)
        sampled_depth = depth_map[v_int, u_int]
        
        # 修复: 从预计算的VGGT深度图中采样 - 考虑图像尺寸调整
        vggt_depth_map = vggt_depths[view_idx].to(device)
        
        # 计算VGGT深度图的缩放因子
        H, W = images[view_idx].shape[1], images[view_idx].shape[2]
        patch_size = 14
        new_H = ((H + patch_size - 1) // patch_size) * patch_size
        new_W = ((W + patch_size - 1) // patch_size) * patch_size
        
        scale_x = new_W / W
        scale_y = new_H / H
        
        # 调整坐标到VGGT深度图尺寸
        u_vggt = u * scale_x
        v_vggt = v * scale_y
        
        u_int_vggt = torch.clamp(torch.tensor(u_vggt).long(), 0, vggt_depth_map.shape[1] - 1)
        v_int_vggt = torch.clamp(torch.tensor(v_vggt).long(), 0, vggt_depth_map.shape[0] - 1)
        vggt_depth = vggt_depth_map[v_int_vggt, u_int_vggt].item()
        
        # 深度一致性检查
        depth_diff = abs(point_cam[2].item() - sampled_depth.item())
        vggt_depth_diff = abs(point_cam[2].item() - vggt_depth)
        
        # 动态容差 - 放宽容差
        dynamic_tolerance = 0.5 * (1 + point_cam[2].item() / 5.0)  # 增加容差
        
        original_consistent = depth_diff < dynamic_tolerance
        vggt_consistent = vggt_depth_diff < dynamic_tolerance * 3.0  # 给VGGT更多容差
        
        projected_points.append((u, v, point_cam[2].item()))
        depth_consistencies.append((original_consistent, vggt_consistent, vggt_depth))
    
    # 判断是否应该增强 - 降低支持率要求
    if len(projected_points) < 2:
        return False, [point], [color]
    
    vggt_support_count = sum(1 for _, vggt_consistent, _ in depth_consistencies if vggt_consistent)
    if vggt_support_count < len(projected_points) * 0.2:  # 只需要20%支持率
        return False, [point], [color]
    
    # 生成增强点
    enhancement_points = [point]
    enhancement_colors = [color]
    
    # 基于VGGT深度生成增强点
    for i, (u, v, original_depth) in enumerate(projected_points):
        _, vggt_consistent, vggt_depth = depth_consistencies[i]
        
        if vggt_consistent and vggt_depth is not None:
            # 增加更多深度偏移
            depth_offsets = [-0.05, -0.02, 0.02, 0.05, 0.1]
            
            for offset in depth_offsets:
                new_depth = vggt_depth + offset
                enhanced_point = backproject_point(u, v, new_depth, Ks[i], Ts[i])
                
                if enhanced_point is not None:
                    enhanced_color = color.copy()
                    color_enhancement = 1.0 + 0.1 * vggt_enhancement_factor
                    enhanced_color = np.clip(enhanced_color * color_enhancement, 0, 1)
                    
                    enhancement_points.append(enhanced_point)
                    enhancement_colors.append(enhanced_color)

    # 添加调试信息
    # if len(projected_points) > 0:
    #     print(f"点 {i}: 投影到 {len(projected_points)} 个视角")
    #     print(f"  VGGT支持: {vggt_support_count}/{len(projected_points)}")
    #     print(f"  深度差异: {[abs(pc[2] - vd) for pc, (_, _, vd) in zip(projected_points, depth_consistencies)]}")
    #     print(f"  容差: {dynamic_tolerance:.3f}")
    #     print(f"  坐标调整: u={u:.1f}->{u_vggt:.1f}, v={v:.1f}->{v_vggt:.1f}")
    
    return True, enhancement_points, enhancement_colors

def apply_vggt_depth_enhancement(base_pcd, scene_info, depth_maps, quality_scores, model, device,
                               vggt_quality_threshold=0.3, vggt_enhancement_factor=1.5,
                               vggt_depths=None):
    """阶段2: VGGT深度增强（使用预计算的深度图）"""
    print("=== 阶段2: VGGT深度增强（使用预计算的深度图） ===")
    
    base_points = np.asarray(base_pcd.points)
    base_colors = np.asarray(base_pcd.colors)
    
    print(f"基础点云包含 {len(base_points)} 个点")
    
    # 优先复用质量评估阶段的深度；否则逐张推理（不再整批 stack）
    if vggt_depths is None:
        torch.cuda.empty_cache()
        vggt_depths = precompute_vggt_depths(scene_info.images, model, device)
    else:
        print(f"复用已有 VGGT 深度图，形状: {tuple(vggt_depths.shape)}")
    
    avg_quality = np.mean(quality_scores)
    print(f"平均VGGT质量分数: {avg_quality:.3f}")
    
    effective_threshold = min(vggt_quality_threshold, 0.3)
    if avg_quality < effective_threshold:
        print(f"VGGT质量不足，跳过增强")
        return base_pcd
    
    # 生成增强点云
    enhanced_points = []
    enhanced_colors = []
    
    print("开始VGGT深度增强处理...")
    for i, point in enumerate(base_points):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(base_points)} ({i/len(base_points)*100:.1f}%)")
        
        should_enhance, enhancement_points, enhancement_colors = check_and_generate_vggt_enhancement(
            point, base_colors[i], scene_info, depth_maps, vggt_depths, quality_scores, model, device,
            vggt_quality_threshold, vggt_enhancement_factor
        )
        
        if should_enhance:
            enhanced_points.extend(enhancement_points)
            enhanced_colors.extend(enhancement_colors)
        else:
            enhanced_points.append(point)
            enhanced_colors.append(base_colors[i])
    
    # 创建增强后的点云
    enhanced_pcd = o3d.geometry.PointCloud()
    enhanced_pcd.points = o3d.utility.Vector3dVector(np.array(enhanced_points))
    enhanced_pcd.colors = o3d.utility.Vector3dVector(np.array(enhanced_colors))
    
    print(f"VGGT深度增强完成，最终点云包含 {len(enhanced_points)} 个点")
    
    return enhanced_pcd

# ==================== 阶段3: 增强点云合并 ====================
def get_visual_hull_with_vggt_enhancement(N, bbox, scene_info, cam_center, depth_maps, quality_scores, model, device,
                                         vggt_quality_threshold=0.3, vggt_enhancement_factor=1.5,
                                         vggt_depths=None):
    """阶段3: 增强点云合并"""
    print("=== 阶段3: 增强点云合并 ===")
    
    # 阶段1: 基础视觉体生成
    base_pcd, base_bbox = get_visual_hull_original(N, bbox, scene_info, cam_center, device)
    
    if base_pcd is None:
        return None, None
    
    # 阶段2: VGGT深度增强
    enhanced_pcd = apply_vggt_depth_enhancement(
        base_pcd, scene_info, depth_maps, quality_scores, model, device,
        vggt_quality_threshold, vggt_enhancement_factor, vggt_depths=vggt_depths
    )
    
    print(f"=== 最终结果 ===")
    print(f"基础点云点数: {len(np.asarray(base_pcd.points))}")
    print(f"增强后点云点数: {len(np.asarray(enhanced_pcd.points))}")
    print(f"点云增长: {len(np.asarray(enhanced_pcd.points)) - len(np.asarray(base_pcd.points))} 个点")
    
    return enhanced_pcd, enhanced_pcd.get_axis_aligned_bounding_box()


def main():
    parser = argparse.ArgumentParser(description='VGGT深度增强的视觉外壳生成')
    parser.add_argument('--data_dir', type=str, default='sparse_nerf_datasets/sparse_omni3d_undistorted/backpack_016', help='数据目录')
    parser.add_argument('--model_path', type=str, default=os.path.join(_VGGT_DIR, 'models', 'model.pt'), help='VGGT模型路径')
    parser.add_argument('--sparse_id', type=int, default=4, help='稀疏视角ID')
    parser.add_argument('--reso', type=int, default=1, help='图像分辨率')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--not_vis', action='store_true', help='不显示可视化')
    parser.add_argument('--cube_size', type=float, default=4.0, help='立方体大小（米）')
    parser.add_argument('--voxel_num', type=int, default=200, help='体素数量')
    
    # VGGT参数
    parser.add_argument('--vggt_quality_threshold', type=float, default=0.3, 
                       help='VGGT质量阈值 (0.0-1.0)，越高VGGT参与度越低')
    parser.add_argument('--vggt_enhancement_factor', type=float, default=1.5, 
                       help='VGGT增强因子 (0.0-2.0)，越大VGGT增强效果越明显')
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    # 加载VGGT模型
    print("加载VGGT模型...")
    try:
        model = VGGT()
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("VGGT模型加载成功")
    except Exception as e:
        print(f"VGGT模型加载失败: {e}")
        return
    
    # 加载稀疏视角配置
    sparse_file = os.path.join(args.data_dir, f"sparse_{args.sparse_id}.txt")
    if not os.path.exists(sparse_file):
        print(f"错误: 稀疏视角文件不存在 {sparse_file}")
        return
    
    selected_ids = np.loadtxt(sparse_file, dtype=np.int32)
    print(f"使用稀疏视角 {args.sparse_id}: {len(selected_ids)} 个视角")
    
    # 使用GaussianObject的加载器
    extra_opts = Namespace()
    extra_opts.sparse_view_num = -1
    extra_opts.resolution = args.reso
    extra_opts.use_mask = True
    extra_opts.data_device = 'cuda'
    extra_opts.init_pcd_name = 'origin'
    extra_opts.white_background = False
    
    # 加载场景信息
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.data_dir, 'images', False, extra_opts=extra_opts)
    camlist = cameraList_from_camInfos(scene_info.train_cameras, 1.0, extra_opts)
    
    # 根据selected_ids筛选相机
    selected_cameras = []
    for idx in selected_ids:
        if idx < len(camlist):
            selected_cameras.append(camlist[idx])
        else:
            print(f"警告: 视角索引 {idx} 超出范围")
    
    # 提取相机参数
    Ks = []
    Ts = []
    images = []
    masks = []
    
    for i, cam_info in enumerate(selected_cameras):
        # 内参
        fx = fov2focal(cam_info.FoVx, cam_info.image_width)
        fy = fov2focal(cam_info.FoVy, cam_info.image_height)
        Ks.append(torch.tensor([[fx, 0, cam_info.image_width/2], [0, fy, cam_info.image_height/2], [0, 0, 1]]))
        
        # 外参
        Ts.append(cam_info.world_view_transform.T)
        
        # 图像和掩码
        images.append(cam_info.original_image)
        masks.append(cam_info.mask)
    
    # 创建SceneInfo对象
    class SceneInfo(NamedTuple):
        Ks: list
        Ts: list
        images: list
        masks: list
    
    scene_info = SceneInfo(Ks, Ts, images, masks)
    
    # 计算相机中心
    cam_locations = []
    for cam_info in selected_cameras:
        cam_locations.append(cam_info.camera_center)
    cam_center = torch.stack(cam_locations).mean(0)
    print(f"计算得到的相机中心: {cam_center}")
    
    # 加载现有深度图
    depth_maps = load_existing_depth_maps(args.data_dir, selected_ids, (images[0].shape[1], images[0].shape[2]))
    
    # 使用VGGT评估深度图质量（同时得到逐视角深度，后续增强直接复用）
    quality_scores, vggt_depths = evaluate_depth_quality_with_vggt(images, depth_maps, model, device)
    torch.cuda.empty_cache()
    
    # 初始化边界框
    bx = args.cube_size
    init_bbox = [[cam_center[0]-bx, cam_center[1]-bx, cam_center[2]-bx], 
                 [cam_center[0]+bx, cam_center[1]+bx, cam_center[2]+bx]]
    
    # 生成VGGT增强的视觉外壳
    pcd, bbox = get_visual_hull_with_vggt_enhancement(
        args.voxel_num, init_bbox, scene_info, cam_center, depth_maps, quality_scores, model, device,
        vggt_quality_threshold=args.vggt_quality_threshold,
        vggt_enhancement_factor=args.vggt_enhancement_factor,
        vggt_depths=vggt_depths,
    )
    
    # 重新计算边界框并调整大小
    bbox_min = bbox.get_min_bound()
    bbox_max = bbox.get_max_bound()
    center = (bbox_min + bbox_max) / 2
    extents = bbox_max - bbox_min
    scale_factor = 2
    scaled_extents = extents * scale_factor
    enlarged_bbox_min = center - scaled_extents / 2
    enlarged_bbox_max = center + scaled_extents / 2

    # 重新生成点云并保存（复用同一份 VGGT 深度，不再二次推理）
    pcd, bbox_new = get_visual_hull_with_vggt_enhancement(
        64, [enlarged_bbox_min, enlarged_bbox_max], scene_info, [0,0,0], depth_maps, quality_scores, model, device,
        vggt_quality_threshold=args.vggt_quality_threshold,
        vggt_enhancement_factor=args.vggt_enhancement_factor,
        vggt_depths=vggt_depths,
    )
    
    # 保存点云
    output_file = os.path.join(args.output_dir, f"visual_hull_vggt_enhanced_{args.sparse_id}.ply")
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"点云已保存到: {output_file}")
    
    # 可视化
    if not args.not_vis:
        print("显示点云...")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()

