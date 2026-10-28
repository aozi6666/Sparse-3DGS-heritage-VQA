# 工作 A · 稀疏 3DGS 指标归档

与 [`VQA/eval_results/`](../../VQA/eval_results/) 对称：**小体积 CSV 进 Git**，大产物仍在 ignore 的 `**/output/`。

## 文件

| 文件 | 说明 |
|------|------|
| [`lineB_kitchen_mip360.csv`](lineB_kitchen_mip360.csv) | mip360/kitchen · 主表（2×2 / V2 / repair / CADR / P2 / P3 depthft-repair） |
| [`lineB_garden_mip360.csv`](lineB_garden_mip360.csv) | mip360/garden · classic / orig-VGGT / depthft（P2+P3） |
| [`lineB_kitchen_iter_sweep.csv`](lineB_kitchen_iter_sweep.csv) | kitchen PSNR–iter（P3；跑脚本后生成） |
| [`ply_init_stats_kitchen.csv`](ply_init_stats_kitchen.csv) | kitchen init ply 点数 / bbox |
| [`SUMMARY_coarse_2x2.md`](SUMMARY_coarse_2x2.md) | 粗训消融文字摘要 |
| [`SUMMARY_repair_cadr.md`](SUMMARY_repair_cadr.md) | repair/CADR/底模文字摘要 |
| [`SUMMARY_depthft_expand.md`](SUMMARY_depthft_expand.md) | depthft_mip360 扩展（P3） |

## 协议

- 场景：`mip360/kitchen`，`resolution=4`，`sparse_view_num=9`，`sh_degree=2`
- 粗训：`iterations=10000` → `ours_10000`
- repair 主对照：RV5.1 LoRA ± CADR；abs16+CADR 附录

---

## 摘要 · 粗训 2×2（test，2026-10-28 齐）

计划：[`docs/PLAN_g0_drop_vggt_ablation.md`](../../docs/PLAN_g0_drop_vggt_ablation.md)

| | 无 Drop | Drop |
|--|--------:|-----:|
| classic | **27.11** (`kitchen_classic`) | 27.10 (`kitchen_drop_classic`) |
| VGGT-enhanced | 26.05 (`kitchen`) | 26.14 (`kitchen_drop`) |

| 效应 | dPSNR | 解读 |
|------|------:|------|
| VGGT−classic @无Drop | **−1.07** | init 主效应为负 |
| VGGT−classic @Drop | **−0.96** | 同上，稳定 |
| Drop−noDrop @classic | **−0.01** | 持平 / 噪声 |
| Drop−noDrop @VGGT | **+0.09** | Drop 弱正 |
| 交互 (VGGT 劣势在 Drop 下变化) | ≈**+0.10** | Drop 略缩小 VGGT 劣势 |

**结论：** 粗训由 **init** 主导（classic ≫ VGGT）；Drop 次要。VGGT 增强 **不宜当粗训主涨点**。

### ply 定性（与 −1 dB 同向）

| init | n_points | bbox 边长量级 |
|------|---------:|---------------|
| `visual_hull_4` | **8616** | ~1.52 / 1.90 / 1.80 |
| `visual_hull_vggt_enhanced_4` | **2874** | ~3.92 / 3.76 / 3.68（更大更稀） |

---

## 摘要 · V2 纯 VGGT + depthft（负对照，已跑）

计划：[`docs/PLAN_vggt_pure_depthft.md`](../../docs/PLAN_vggt_pure_depthft.md)  
目录：`kitchen_drop_vggt_pure_depthft` / `visual_hull_vggt_pure_depthft_4`

| | PSNR | SSIM | LPIPS |
|--|-----:|-----:|------:|
| V2 pure+depthft (test) | **11.78** | 0.8546 | 0.1600 |
| vs `kitchen_drop` | **−14.36** | −0.092 | +0.115 |

**解读：** 崩盘合理 → 缺 classic hull / COLMAP **度量对齐** 时，单目深度反投影点云错位；作 **负对照**（不能扔掉 visual hull），勿当主方法。若再试须先尺度对齐。

---

## 摘要 · 主表 C0–C4 / repair / CADR

| ID | 含义 | test 要点 |
|----|------|-----------|
| **C0** | classic 粗训 | PSNR≈**27.11**（粗训最优） |
| **C1** | SD1.5 repair | vs 旧 VGGT 粗训 dPSNR **+0.80** |
| **C2 / A0** | RV51 无 CADR | vs C1 ≈**+0.10**（无实质涨点） |
| **C3 / A3** | RV51 + 完整 CADR | vs C2 ≈**−0.003**（持平；目标 +0.2 未达） |
| **C4** | abs16 无 CADR | **26.928**（vs A0 −0.038） |

### CADR 分量（A-full，已归档）

详见 [`SUMMARY_repair_cadr.md`](SUMMARY_repair_cadr.md)。A2/A4 ≈持平微正；A5 −0.11（默认关）；目标 +0.2 未达。

---

## P2 已完成

| 项 | 结果 |
|----|------|
| classic→RV51 repair | test 27.085（vs VGGT-RV51 +0.12） |
| garden classic vs VGGT | 33.49 / 33.37（dPSNR −0.11） |

## P3 已完成

详见 [`SUMMARY_depthft_expand.md`](SUMMARY_depthft_expand.md) · [`SUMMARY_coarse_2x2.md`](SUMMARY_coarse_2x2.md)（iter 表）。

| 项 | 结果（test） |
|----|------|
| PSNR–iter | classic≫VGGT 全程；10k dPSNR≈−1.53 |
| depthft→RV51 / +CADR | 27.02 / 27.07 |
| garden depthft | 33.53（≥ classic / orig VGGT） |

**工作 A 主消融已齐。** 可选仅剩：V2 尺度对齐（非必须）；论文作图。

## 主对照跑法（RV51 ± CADR）

```bash
cd 3DGS/GaussianObject/sh/lineB_gaussianobject
SD_CKPT=./models/Realistic_Vision_V5.1.safetensors \
LORA_EXP=controlnet_finetune/kitchen_drop_rv51 \
BASE_TAG=kitchen_drop_rv51 \
USE_DROP=1 SKIP_PATH=1 bash 0_run_all.sh 8
```
