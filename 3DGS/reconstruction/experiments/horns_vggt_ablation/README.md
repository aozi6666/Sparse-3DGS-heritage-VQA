# horns：九视角 SfM / SfM＋冻结 VGGT 初始化消融

## 隔离与运行

仅增加 `experiments/horns_vggt_ablation/`，不修改既有训练、深度和准备模块。
输入 `data/LLFF/horns/` 只读。正式输出 `output/horns_vggt9_v1/`；日志在其 `logs/` 下。
入口自动激活项目 `.venv-ddgs-vggt`。CPU预检无需导入PyTorch；GPU实例内存至少12 GiB，建议16 GiB以上。

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
bash 3DGS/reconstruction/experiments/horns_vggt_ablation/start.sh --stage check
nohup bash 3DGS/reconstruction/experiments/horns_vggt_ablation/start.sh --stage run > 3DGS/reconstruction/output/horns_vggt9_v1.console.log 2>&1 < /dev/null &
tail -f 3DGS/reconstruction/output/horns_vggt9_v1.console.log
```

总日志显示当前阶段；训练详细进度：`tail -f 3DGS/reconstruction/output/horns_vggt9_v1/logs/A.log`（B同理）。
相同命令重跑跳过有成功标记的阶段；训练中断后未完成的训练阶段从头训练，不是优化器断点续训。
配置、原图、相机文件、权重及关键代码均保存哈希。配置改变需新 `--out`。
准备阶段中断保留现场，需新 `--out`；不会清理用户已有目录。进程锁阻止同一输出目录并行启动。

## 公平性与边界

62张已标定照片排序后，每八张留出一张，共8张测试；从剩余54张均匀选9张训练，另外45张不参与RGB训练。
展示相机为测试列表中间一张，训练前固定，保存于split.json。
串行去畸变，更新相机和二维观测，生成1008×756图像，不另存多套缩略图。
基础点云：复用原有全序列相机位姿和特征轨迹关联，只使用9张训练图的二维观测重新DLT三角化；丢弃原始XYZ和测试观测。筛选阈值：重投影误差<=2像素（缩小后尺寸），三角化角度>=1度。
**这是已知位姿、已知关联条件下的初始化消融，不是只输入9张原图完成独立标定与匹配的实验。**

A使用基础点云。B使用完全相同的基础点云＋冻结VGGT补点。VGGT采用现有脚本的单帧深度预测，使用9张训练图的三角化点对齐尺度，用训练图间的深度一致性筛选补点；不是9帧联合VGGT预测。预测列表中的相机只有9张训练图，不能调用全62张推理。
两组均保留真实背景；同种子0、同10000步、同分辨率、相同加密规则；关闭Drop、LoRA、repair和深度损失。不使用Visual Hull，不代表完整SynGS。
VGGT补点若筛选后为0，直接报错，不能把没有实际处理差异的实验当作有效对照。

## 输出

- `A_init_sfm.ply`、`B_init_sfm_vggt.ply`：两份可打开的初始化点云。
- `supplementation.json`：基础点、新增点、各训练视角候选／接受点数。
- `point_counts.json`：两份初始化点云点数。
- `A/`、`B/`：只保存最后一步高斯模型及配置，不保存优化器、TensorBoard图片或中间模型。
- `visualization/A.png`、`B.png`、`GT.png`：同一留出视角、完整背景、原始导出分辨率。
- `visualization/A_labeled.jpg`、`B_labeled.jpg`、`comparison.jpg`：带逐图指标的展示版；指标在画面下方，未修改原始画面。
- `visualization/A_points.png`、`B_points.png`：同一三维观察角、坐标范围、点大小，显示最多60000点但标注完整点数；保留PLY用于核对。不是精确表面误差图。
- `per_view.csv`：8个测试视角各自的PSNR、SSIM、LPIPS及LPIPS×100。
- `comparison_metrics.csv`、`test_mean_metrics.json`：测试视角等权平均指标、初始化点数、最终高斯数及B−A差值。
- `complete.json`：所有结果成功导出后的完成标记。

所有RGB指标比较8位导出结果和相同量化GT；整幅画面含背景。LPIPS采用VGG、normalize=True；LPIPS*为原值×100。
**单幅展示指标不等于测试集均值；本实验也不是多场景均值。不预设必须提升，不筛出有利视角、不凑差距。**
点数增加仅代表补点发生，性能提升须由测试指标及图像共同判断。最终高斯点数受加密／剪枝影响。

## 省盘和已验证范围

共享一份去畸变后的1/4图像。训练深度目录通过符号链接共享RGB；只缓存9张深度／置信度图。无需重复下载模型，不运行修复流程。运行前至少5 GiB可用空间，各阶段启动前低于3 GiB则停止。
CPU集成测试完成：62张图去畸变、9/8视角划分、DLT、train-only COLMAP重建读写；得到12866个基础点；每张训练图有3336–6803个尺度锚点。测试目录为output/horns_vggt9_cpucheck，仅为独立验证结果。
GPU阶段（VGGT推理、两次训练、最终渲染与评价）尚未实际运行。质量结论待真实实验结果。
