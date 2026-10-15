# VGGT depth_head 域适应 · 粗训对照归档（负结果）

与 [`lineB_kitchen_mip360.csv`](lineB_kitchen_mip360.csv) **分开存放**（不改旧 CADR/LoRA 主表）。  
根 README 摘要：[`../../README.md`](../../README.md) → 工作 A ·「VGGT depth_head 域适应（负结果）」。  
历史设计稿：[`../../docs/PLAN_vggt_dpt_depth_adapt.md`](../../docs/PLAN_vggt_dpt_depth_adapt.md)（已归档状态；方法未作论文主贡献）。

## 文件

| 文件 | 说明 |
|------|------|
| [`lineB_kitchen_vggt_depthft.csv`](lineB_kitchen_vggt_depthft.csv) | G1 vs depthft 粗训 absolute/delta |
| [`depthft_scale_diag_kitchen.csv`](depthft_scale_diag_kitchen.csv) | 尺度诊断逐帧（40×base + 40×depthft） |
| [`depthft_scale_diag_kitchen_summary.csv`](depthft_scale_diag_kitchen_summary.csv) | 尺度诊断摘要表 |

## 协议（本表）

- 场景：`mip360/kitchen`，`resolution=4`，`sparse_view_num=9`，`sh_degree=2`
- 粗训：`USE_DROP=1`，`iterations=10000` → `ours_10000`
- 门控：默认 **GATE_ZOE**（Zoe–VGGT 一致性；与 G1 相同）
- 加载：`model.pt`（base）+ `depth_head_ft_*.pt`（overlay）；**不覆盖** `model.pt`
- 成功标准（原计划）：相对 G1 粗训 test **PSNR ≥ +0.15** → **未达**

## 摘要（test · ours_10000 · 2026-10-15）

| ID | 设定 | init ply | depth_head | PSNR | SSIM | LPIPS | ΔPSNR vs G1 |
|----|------|----------|------------|-----:|-----:|------:|------------:|
| G1 | 原 VGGT | `visual_hull_vggt_enhanced_4` | — | **26.141** | 0.9468 | 0.0454 | — |
| D1-kitchen | kitchen-only ft | `visual_hull_vggt_depthft_4` | `depth_head_ft_kitchen.pt` | 26.088 | 0.9464 | 0.0465 | **−0.053** |
| D1-mip360 | kitchen+bonsai+garden ft | `visual_hull_vggt_depthft_mip360_4` | `depth_head_ft_mip360.pt` | 25.978 | 0.9460 | 0.0462 | **−0.163** |
| D1-holdout | bonsai+garden ft（kitchen 零样本） | — | — | — | — | — | **SKIPPED** |

**结论（粗训）：** 两版 depthft 均未达 +0.15；相对 G1 略降（mip360 **ΔPSNR −0.163**）。**不作为论文主贡献**；权重 / 代码 / CSV 保留。

## 尺度诊断（kitchen · 40 帧 · 2026-10-15）

| tag | median_scale | std | aligned_L1 |
|-----|-------------:|----:|-----------:|
| base | 4.09 | 0.42 | 0.051 |
| depthft_mip360 | 4.07 | 0.41 | 0.052 |

1. **scale≈4**：VGGT↔COLMAP 系统尺度差（非坏掉）；训练用 `median_scale_l1` 吸收，未改 loss。  
2. **微调≈无效**：aligned_L1 几乎不变 → 解释粗训不涨。  
3. **hold-out**：已跳过（证据已足够）。

## 产物路径（均保留）

| 角色 | 路径 |
|------|------|
| G1 metrics | `GaussianObject/output/gs_init/kitchen_drop/test/ours_10000/results.json` |
| kitchen-only | `.../kitchen_drop_depthft/test/ours_10000/results.json` |
| mip360 三场景 | `.../kitchen_drop_depthft_mip360/test/ours_10000/results.json` |
| 权重 | `3DGS/vggt/models/depth_head_ft_kitchen.pt`、`depth_head_ft_mip360.pt` |
| 代码 | `tools/finetune_depth_head_colmap.py`、`sh/lineB_gaussianobject/*depthft*` |
