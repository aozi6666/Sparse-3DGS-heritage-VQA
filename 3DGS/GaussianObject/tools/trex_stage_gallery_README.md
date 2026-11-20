# trex 带真实背景的训练阶段可视化

服务器入口：`3DGS/GaussianObject/tools/run_trex_stage_gallery.sh`

- 输入：`data/LLFF/trex_fullscene` 已去畸变的55张照片；复用相机与SfM点云。
- 固定视角：`DJI_20200223_163616_140`，分辨率为原标定尺寸的1/4。
- 全量RGB训练10000步，不使用VGGT、Drop、LoRA、repair；非完整SynGS实验。
- 每100步从1000步起采图；候选PNG仅存在内存，不写入磁盘。
- 按实测PSNR选择九张不同阶段图，目标最高与最低相差0.5–2 dB；不保证SSIM/LPIPS同时达到指定跨度。不满足目标时仍导出九张，并将target_met置为false。
- 所有指标针对实际导出的8位PNG与相同量化GT、整幅画面含背景计算。LPIPS采用VGG、normalize=True。CSV提供LPIPS及LPIPS_x100。
- 只保存一个最终模型；不保存优化器检查点或TensorBoard图片。不删除用户已有文件。高斯数量阈值是停止加密的软阈值，最后一次加密可能超出。
- 输出目录：`output/trex_stage_gallery_v1/trex`，含9张渲染PNG、GT.png、overview.jpg、metrics.csv、candidates.json、complete.json、model/。
- 九张图属于训练视角的训练阶段展示，不能标注为九种方法，不能代替多场景平均表或留出视角评测。
- 运行前至少4 GiB可用空间；训练采图期间低于2 GiB会中止。部分运行目录保留；需要重跑时使用新的RUN_TAG，不能当作断点续训。

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
bash 3DGS/GaussianObject/tools/run_trex_stage_gallery.sh --check
nohup bash 3DGS/GaussianObject/tools/run_trex_stage_gallery.sh > 3DGS/GaussianObject/output/trex_stage_gallery_v1.console.log 2>&1 < /dev/null &
tail -f 3DGS/GaussianObject/output/trex_stage_gallery_v1.console.log
```

入口自动激活`.venv-ddgs-vggt`。CPU预检已通过；GPU训练和渲染待用户启用GPU后运行。
