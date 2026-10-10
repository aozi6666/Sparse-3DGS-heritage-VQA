# 工作 A · 稀疏 3DGS 指标归档

与 [`VQA/eval_results/`](../../VQA/eval_results/) 对称：**小体积 CSV 进 Git**，大产物仍在 ignore 的 `**/output/`。

## 文件

| 文件 | 说明 |
|------|------|
| [`lineB_kitchen_mip360.csv`](lineB_kitchen_mip360.csv) | mip360/kitchen · lineB |

## 协议（本表）

- 场景：`mip360/kitchen`，`resolution=4`，`sparse_view_num=9`，`sh_degree=2`
- 初始化：`visual_hull_vggt_enhanced_4`（VGGT）
- 粗训：`USE_DROP=1`，`iterations=10000` → `ours_10000`
- **主线修复**：`Realistic_Vision_V5.1.safetensors` + LoRA `rank=32` → `kitchen_drop_rv51` → `ours_None__kitchen_drop_rv51`
- 历史对照：曾用官方 SD1.5 ckpt → `ours_None`（CSV 中仍保留）

## 摘要

### LoRA repair vs 粗训（test，历史 SD1.5 底座）

| | dPSNR | dSSIM | dLPIPS |
|--|------:|------:|-------:|
| repair vs coarse | **+0.80** | +0.0026 | −0.0063 |

### RV51 vs 历史 SD1.5 repair（test，同设定）

| | PSNR | SSIM | LPIPS |
|--|-----:|-----:|------:|
| SD1.5 repair | 26.86 | 0.9493 | 0.0394 |
| RV51 repair | 26.97 | 0.9494 | 0.0399 |
| Δ | **+0.10** | ≈0 | +0.0006（略差） |

结论：换 RV51 **无实质涨点**；主线仍用 RV51（差异化底模），指标以 CSV 为准。

### 待补

| 对照 | 说明 |
|------|------|
| VGGT vs classic hull | 见 CSV `delta_pending` |
| Drop 消融 | `2b_train_gs_drop_ablation.sh` |
| 步数 sweep | 见根 [`README.md`](../../README.md) TODO |

## 默认跑法

```bash
cd 3DGS/GaussianObject/sh/lineB_gaussianobject
USE_DROP=1 SKIP_PATH=1 bash 0_run_all.sh 7   # 默认即 RV51
```
