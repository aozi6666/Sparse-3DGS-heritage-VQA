# CADR A-full + abs16 无 CADR（C4）+ 主表 C0–C3

> 状态：脚本已落地；**训练由你本地执行**（结果-only 省盘）。  
> 分支：`feat/sd15-stronger-base-lora` · 环境：`.venv-ddgs-vggt`  
> 编排：[`run_cadr_ablation_a2a4a5_c4.sh`](../3DGS/GaussianObject/sh/lineB_gaussianobject/run_cadr_ablation_a2a4a5_c4.sh)  
> 指标：[`lineB_kitchen_mip360.csv`](../3DGS/eval_results/lineB_kitchen_mip360.csv)

## 1. 主表编号（P1，不必重训粗训）

| ID | 含义 | 目录 / tag | 状态 |
|----|------|------------|------|
| **C0** | 粗训最优（classic） | `kitchen_classic` / `kitchen_drop_classic` | 已有 |
| **C1** | SD1.5 repair | `ours_None` | 已有 |
| **C2 / A0** | RV51 repair 无 CADR | `ours_None__kitchen_drop_rv51` | 已有 |
| **C3 / A3** | RV51 + 完整 CADR | `ours_None__kitchen_drop_rv51_conf` | 已有 |
| **C4** | abs16 无 CADR | `ours_None__kitchen_drop_abs16` | **本脚本新跑** |

叙事：粗训最优是 **classic**；repair 主对照仍在 `kitchen_drop` init + RV51±CADR 上（与历史行公平）。

## 2. CADR 分量（A-full）

同 RV51 LoRA、`GS_DIR=kitchen_drop`，只改步骤 8：

| ID | tag | 设定 |
|----|-----|------|
| A2 | `kitchen_drop_rv51_trust` | `conf_trust_only`（need≡1） |
| A3 | `kitchen_drop_rv51_conf` | 完整 CADR（**不重训**） |
| A4 | `kitchen_drop_rv51_cov` | CADR + `use_cov_in_need` |
| A5 | `kitchen_drop_rv51_aden` | CADR + `use_adaptive_denoise` |

## 3. 省盘（只要结果不要日志）

- 默认 `save_controlnet_preview=False`（不写 `controlnet_out/it*.png`）
- 每档结束后删该 tag 的 `tb_logs/`、`csv_logs/`、`save/controlnet_out/`
- 保留 `save/last.ply`、`configs/`、`gs_init/.../ours_None__<tag>/results.json`

## 4. 本地执行

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
git switch feat/sd15-stronger-base-lora
source .venv-ddgs-vggt/bin/activate
export CUDA_VISIBLE_DEVICES=0 SKIP_PATH=1
cd 3DGS/GaussianObject/sh/lineB_gaussianobject
bash run_cadr_ablation_a2a4a5_c4.sh
# 断点：bash run_cadr_ablation_a2a4a5_c4.sh 2
```

| stage | 动作 |
|-------|------|
| 0 | D1 清旧 repair 日志 |
| 1 | A2 trust-only |
| 2 | A4 cov→need |
| 3 | A5 adaptive denoise |
| 4 | C4 abs16 无 CADR |
| 5 | 写 CSV |

## 5. 明确不做

不重训 LoRA/粗训；不对 V2 做尺度对齐重跑；不跑步数 sweep。
