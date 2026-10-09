# 系统架构（占位）

本系统目标：受限采集下重建可漫游的 **3DGS 数字资产**，并与现场视频的 **理解 / 问答 / 空间定位** 融合（详见仓库根目录外的 `毕设.md` 设想）。

```text
受限采集视频/图像
        │
        ├─► 工作 A：稀疏 3DGS 重建  ──►  Gaussian Scene（可渲染资产）
        │
        └─► 工作 B：视频理解 + 定位 ──►  语义回答 + 当前视角/区域
                        │
                        └─► 融合层：问答 / 导览提示 / 3D 漫游
```

## 工作 A（本仓已实现主线）

- 输入：稀疏视角 + 相机位姿 + VGGT/visual hull 初始点云  
- 训练：GaussianObject `train_gs` + DropGaussian（opacity dropout）  
- 可选提质：LOO → LoRA → repair  
- 输出：`output/**/point_cloud` / `last.ply`，以及 `results.json` 指标  

入口：`GaussianObject/sh/lineB_gaussianobject/0_run_all.sh`。

## 工作 B（后续接入）

- 占位目录：[`../../VQA/`](../../VQA/)  
- 候选代码：仓库同级 `Mage/`、`VLMEvalKit/`（尚未迁入本仓）  
- 与工作 A 的接口预期：共享场景坐标系；查询帧 ↔ 3DGS 位姿/区域；渲染服务提供当前/邻近视角  

## 本阶段不做

不在此文档实现融合调度代码；仅固定「重建资产」与「问答」的边界，便于第二阶段迁入。
