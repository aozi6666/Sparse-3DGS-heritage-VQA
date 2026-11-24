# 独立任务 V2：纯 VGGT + depth_head 微调粗训

> 状态：**已跑通一轮（负对照）**；test PSNR **11.78**（vs `kitchen_drop` ≈−14 dB）。缺 COLMAP 度量对齐时点云错位 → **勿当主方法**；若再试须先尺度对齐。与 2×2 消融脱钩。  

> 分支：`feat/sd15-stronger-base-lora`  
> 环境：`.venv-ddgs-vggt`  
> 编排：[`run_vggt_pure_depthft_coarse.sh`](../3DGS/GaussianObject/sh/lineB_gaussianobject/run_vggt_pure_depthft_coarse.sh)

## 1. 定义

| 代号 | 含义 |
|------|------|
| **V1（pure）** | 不用 classic visual hull 作底；图像 → VGGT 深度反投影 → init ply |
| **V2（本任务）** | V1 + **depth_head 微调权重**（`model.pt` + `depth_head_ft_*.pt`）→ 粗训+评测 |

与线 B 主路径区别：主路径是 **classic hull ⊕ VGGT 增强**（`visual_hull_vggt_enhanced_*`）。  
历史 `kitchen_drop_depthft*` = enhanced + depthft，**不是** pure。

## 2. 隔离命名（禁止覆盖 2×2）

| 项 | 值 |
|----|-----|
| INIT_PCD | `visual_hull_vggt_pure_depthft_${SPARSE_ID}` |
| GS_NAME | `kitchen_drop_vggt_pure_depthft` |
| GS_DIR | `output/gs_init/kitchen_drop_vggt_pure_depthft` |
| 默认权重 | `3DGS/vggt/models/model.pt` + `depth_head_ft_mip360.pt` |

护栏：不得写入 `kitchen_drop` / `kitchen` / `kitchen_drop_classic` / `kitchen_classic`；不得改写 `visual_hull_vggt_enhanced_*.ply`。

## 3. 实现钩子

- [`vggt_visual_hull_enhanced.py`](../3DGS/GaussianObject/vggt_visual_hull_enhanced.py)：`--init_mode pure` + `--output_ply_name`
- [`1_vggt_pure_hull_depthft.sh`](../3DGS/GaussianObject/sh/lineB_gaussianobject/1_vggt_pure_hull_depthft.sh)
- [`run_vggt_pure_depthft_coarse.sh`](../3DGS/GaussianObject/sh/lineB_gaussianobject/run_vggt_pure_depthft_coarse.sh)：stage 0 清盘 → 1 hull → 2 粗训+评测 → 3 写 CSV

## 4. 你本地执行（与 2×2 分开）

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
git switch feat/sd15-stronger-base-lora
source .venv-ddgs-vggt/bin/activate
export CUDA_VISIBLE_DEVICES=0 SKIP_PATH=1
# 可选：VGGT_DEPTH_HEAD=.../depth_head_ft_mip360.pt
cd 3DGS/GaussianObject/sh/lineB_gaussianobject
bash run_vggt_pure_depthft_coarse.sh
# 断点：bash run_vggt_pure_depthft_coarse.sh 2
```

省盘：只存最后一轮 ply；`--not_saveimages`；`PURE_STRIDE=4` 控点数。

## 5. 指标

跑完 stage 3 后写入 [`lineB_kitchen_mip360.csv`](../3DGS/eval_results/lineB_kitchen_mip360.csv)：

- absolute：`coarse_gs_vggt_pure_depthft`
- delta：`pure_depthft_vs_vggt_drop`（相对 `kitchen_drop`）

## 6. 本任务不做

不跑 LOO/LoRA/repair；不做 CADR；不与 `run_ablation_2x2_fill_classic_nodrop.sh` 混跑。
