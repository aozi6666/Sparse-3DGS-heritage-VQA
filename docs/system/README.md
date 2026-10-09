# 系统架构

论文与总体设计见 [`../thesis.md`](../thesis.md)。

```text
受限采集视频/图像
        │
        ├─► 工作 A：稀疏 3DGS 重建  ──►  Gaussian Scene（可渲染资产）
        │         GaussianObject/ + vggt/ + Drop
        │
        └─► 工作 B：Mage-VL 理解 + 评测 ──► 语义回答 / Video-MME 等指标
                  VQA/mage_vl + VQA/VLMEvalKit
                        │
                        └─►（后续）定位对上 + 融合：问答 / 导览 / 3D 漫游
```

## 工作 A（已实现主线）

入口：`GaussianObject/sh/lineB_gaussianobject/0_run_all.sh`。

## 工作 B（本仓已 vendoring）

- 推理/理解代码：`VQA/mage_vl/`（无大权重；权重外置）  
- 评测代码：`VQA/VLMEvalKit/`（含本项目对 Video-MME / config 的修改）  
- 已跑通指标：`VQA/eval_results/`  

## 尚未实现

帧–3DGS 位姿定位（GSVisLoc 类）与融合调度层：仅在 `thesis.md` 中规划，本阶段不落代码。
