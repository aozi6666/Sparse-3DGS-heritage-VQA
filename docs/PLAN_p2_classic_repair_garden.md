# P2：classic→RV51 repair + garden 粗训对照

> 状态：脚本已落地；**训练由你本地执行**。  
> 分支：`feat/sd15-stronger-base-lora` · 环境：`.venv-ddgs-vggt`  
> 不做步数 sweep；V2 尺度对齐不重跑。

## 任务 1：classic+Drop → 复用 RV51 LoRA（无 CADR）

| 项 | 值 |
|----|-----|
| init GS | `kitchen_drop_classic` |
| LoRA | `controlnet_finetune/kitchen_drop_rv51` |
| REPAIR_TAG | `kitchen_drop_classic_rv51` |
| CADR | 关（对齐 A0） |

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
git switch feat/sd15-stronger-base-lora
source .venv-ddgs-vggt/bin/activate
export CUDA_VISIBLE_DEVICES=0 SKIP_PATH=1
cd 3DGS/GaussianObject/sh/lineB_gaussianobject
bash run_classic_rv51_repair.sh
```

对照：test PSNR vs `ours_None__kitchen_drop_rv51`（≈26.97）。结果写入 `lineB_kitchen_mip360.csv`。

## 任务 2：garden classic vs VGGT 粗训

| 格 | GS_NAME | INIT |
|----|---------|------|
| classic+Drop | `garden_drop_classic` | `visual_hull_4` |
| VGGT+Drop | `garden_drop` | `visual_hull_vggt_enhanced_4` |

```bash
bash run_garden_classic_vs_vggt_coarse.sh
# garden images_2 过大易 OOM → 脚本默认 VGGT_RESO=4；续跑：
# bash run_garden_classic_vs_vggt_coarse.sh 2
# 仍失败：SCENE=bonsai bash run_garden_classic_vs_vggt_coarse.sh
```

指标：[`3DGS/eval_results/lineB_garden_mip360.csv`](../3DGS/eval_results/lineB_garden_mip360.csv)（与 kitchen 主表分离）。

## 省盘

repair：`save_controlnet_preview=False`；每档后删 `tb_logs`/`csv_logs`/`controlnet_out`。  
粗训：只存末轮 ply；render 不存图。

## 验收

- `kitchen_drop_classic/test/ours_None__kitchen_drop_classic_rv51/results.json`
- `garden_drop_classic` 与 `garden_drop` 的 `ours_10000/results.json`
- kitchen 基线目录未被覆盖
