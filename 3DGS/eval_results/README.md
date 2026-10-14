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
- **论文主对照**：同一 RV5.1 LoRA（`*_rv51`）上对比有无 CADR  
  - 无 CADR：`kitchen_drop_rv51` → `ours_None__kitchen_drop_rv51`  
  - 有 CADR：`kitchen_drop_rv51_conf` → `ours_None__kitchen_drop_rv51_conf`
- **附录对照**：AbsoluteReality v1.6 + 同底模 LoRA + CADR → `ours_None__kitchen_drop_abs16_conf`（换底模与 CADR 绑在一起，不能单独归因）
- 历史对照：官方 SD1.5 → `ours_None`（CSV 中仍保留）
- 脚本 `_common.sh` 默认可能指向 abs16；**主对照请用环境变量切回 RV51**（见下方跑法）

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

结论：换 RV51 **无实质涨点**。

### CADR vs RV51（test，同 RV51 LoRA；2026-10-14）

| | PSNR | SSIM | LPIPS |
|--|-----:|-----:|------:|
| RV51（无 CADR） | 26.967 | 0.9494 | 0.0399 |
| CADR（`*_rv51_conf`） | 26.963 | 0.9492 | 0.0397 |
| Δ | **−0.003** | −0.0001 | −0.0003 |

结论：**持平**；CADR 设计目标 test PSNR **+0.2** 未达。方法已落地可跑，代码保留。

### abs16+CADR vs RV51（test，附录；2026-10-14）

| | PSNR | SSIM | LPIPS |
|--|-----:|-----:|------:|
| RV51 | 26.967 | 0.9494 | 0.0399 |
| abs16+CADR | 26.868 | 0.9491 | 0.0394 |
| Δ | **−0.10** | −0.0002 | −0.0005 |

结论：换 AbsoluteReality + CADR **≈ 持平/略降**；不能单独归因 CADR。

### 待补

| 对照 | 说明 |
|------|------|
| CADR 消融 A2/A3（可选 A4/A5） | 同 RV51 LoRA，只改步骤 8 |
| abs16 无 CADR（C4） | 隔离「换底模」与 CADR |
| VGGT vs classic hull | 见 CSV `delta_pending` |
| Drop 消融 | `2b_train_gs_drop_ablation.sh` |
| 步数 sweep | 见根 [`README.md`](../../README.md) TODO |

## 主对照跑法（RV51 ± CADR）

```bash
cd 3DGS/GaussianObject/sh/lineB_gaussianobject
# CADR：复用已有 *_rv51 LoRA，从步骤 8 起
SD_CKPT=./models/Realistic_Vision_V5.1.safetensors \
LORA_EXP=controlnet_finetune/kitchen_drop_rv51 \
BASE_TAG=kitchen_drop_rv51 \
USE_DROP=1 SKIP_PATH=1 bash 0_run_all.sh 8
```

附录 abs16（会写新目录 `*_abs16`，不覆盖 RV51）：

```bash
USE_DROP=1 SKIP_PATH=1 bash 0_run_all.sh 7   # 默认 SD_CKPT=absolutereality_v16
```
