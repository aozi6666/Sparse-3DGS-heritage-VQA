# LLFF horns / fortress 带背景训练阶段可视化

- 独立入口：tools/llff_scene_gallery/start.sh；默认串行 horns、fortress。
- 全量已标定照片参与训练（horns 62 / fortress 42），全幅真实背景。没有留出测试视角，指标只用于训练阶段展示。
- 每场景默认10000步基础3DGS粗训；不启用VGGT、Drop、LoRA或repair，不代表完整SynGS。
- 独立 prepared/<scene> 保存1/4分辨率去畸变图与同步更新的PINHOLE内参、特征坐标；不复制database.db、深度、掩膜，不改原始数据和公共训练代码。
- 固定相机：按标定图像名排序的中间一张。运行前CPU预检会显示名字；所有阶段和GT严格共用该相机。
- 每100步（1000步开始）采图到内存；按真实PSNR选9张，尽量满足整组0.5–2 dB跨度。SSIM / LPIPS不保证指定差值；不修改像素来凑指标。
- 真实8位导出图和GT同分辨率、同量化方式，整幅含背景计算PSNR / SSIM / LPIPS；LPIPS为VGG normalize=True，CSV另有LPIPS_x100。
- 无训练阶段模型、最终高斯模型、优化器快照或TensorBoard图片；模型目录仅有初始化点云、相机等小型运行记录。
- 独立子进程逐场景释放显存。新运行至少4 GiB空闲，采图时低于2 GiB停止。
- 不删除已有文件。完成场景会跳过；失败的半成品保留，用新的RUN_TAG重新启动该场景。不是断点续训。
- 输出 output/llff_horns_fortress_gallery_v1/<scene>/ ：9个编号PNG、GT.png、overview.jpg（带指标）、metrics.csv、candidates.json、complete.json、README.md、model/小型记录。
- 这20张图是两个场景的训练阶段展示，不能标为9种对比方法，单张图指标也不能替代多场景平均表。
- --check仅做CPU检查（包含每场景1张原图在内存去畸变），不准备数据、不训练、不渲染。
- 现有预训练VGG权重必须已在本地缓存，不自动下载。

## 命令
```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
bash 3DGS/GaussianObject/tools/llff_scene_gallery/start.sh --check
nohup bash 3DGS/GaussianObject/tools/llff_scene_gallery/start.sh > 3DGS/GaussianObject/output/llff_horns_fortress_gallery_v1.console.log 2>&1 < /dev/null &
tail -f 3DGS/GaussianObject/output/llff_horns_fortress_gallery_v1.console.log
# 单场景详细训练进度（horns完成后才会生成fortress.log）
tail -f 3DGS/GaussianObject/output/llff_horns_fortress_gallery_v1/horns.log
# 若半成品失败，保留原目录并新起一轮：
# RUN_TAG=llff_horns_fortress_gallery_v2 bash 3DGS/GaussianObject/tools/llff_scene_gallery/start.sh --scenes fortress
```
