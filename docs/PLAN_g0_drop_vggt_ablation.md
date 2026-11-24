# G0 / Drop / VGGT 粗训消融（2×2）

> 状态：**2×2 已跑完并归档**（含 `kitchen_classic`、ply、CSV）。  
> 分支：`feat/sd15-stronger-base-lora` · 环境：`.venv-ddgs-vggt`  
> 入口：[`run_ablation_2x2_fill_classic_nodrop.sh`](../3DGS/GaussianObject/sh/lineB_gaussianobject/run_ablation_2x2_fill_classic_nodrop.sh)  
> 指标：[`lineB_kitchen_mip360.csv`](../3DGS/eval_results/lineB_kitchen_mip360.csv) · [`ply_init_stats_kitchen.csv`](../3DGS/eval_results/ply_init_stats_kitchen.csv) · 摘要 [`eval_results/README.md`](../3DGS/eval_results/README.md)

## 1. 2×2 矩阵与 test PSNR

| | 无 Drop | Drop |
|--|--------:|-----:|
| classic | **27.11** `kitchen_classic` | 27.10 `kitchen_drop_classic` |
| VGGT-enhanced | 26.05 `kitchen` | 26.14 `kitchen_drop` |

| 效应 | dPSNR |
|------|------:|
| VGGT−classic @无Drop | −1.07 |
| VGGT−classic @Drop | −0.96 |
| Drop−noDrop @classic | −0.01 |
| Drop−noDrop @VGGT | +0.09 |

**叙事：** init 主导（classic ≫ VGGT）；Drop 弱正/条件性；VGGT 增强不宜当粗训主涨点。

## 2. ply

| init | n_points | bbox |
|------|---------:|------|
| `visual_hull_4` | 8616 | ~1.5–1.9 |
| `visual_hull_vggt_enhanced_4` | 2874 | ~3.7–3.9 |

## 3. 相关：V2 负对照

纯 VGGT+depthft test PSNR **11.78**（≈−14 dB vs `kitchen_drop`）→ 尺度/几何失败负对照。见 [`PLAN_vggt_pure_depthft.md`](PLAN_vggt_pure_depthft.md)。

## 4. 后续（本文件范围外）

CADR A2/A3、abs16 无 CADR、classic repair 翻盘、多场景、步数 sweep → 根 README TODO。
