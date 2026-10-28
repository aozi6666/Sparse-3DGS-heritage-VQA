# GaussianObject 训练脚本

仓库根：`Sparse-3DGS-heritage-VQA/`（本文件位于 `3DGS/GaussianObject/sh/`）。  
系统总览见根 [`README.md`](../../../README.md)。  
现行主线：`lineB_gaussianobject/`（默认 `USE_DROP=1`）。

## 目录

| 目录 | 说明 |
|------|------|
| `lineB_gaussianobject/` | **推荐**：VGGT hull → 粗 GS（±Drop）→ render → LOO / LoRA / repair |
| `_archive/` | 旧脚本备份，勿直接跑 |

已归档 DDGS / lineA 仅在本地 `_archive/`（不进主线文档）。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `USE_DROP` | `1` | DropGaussian；目录用 `kitchen_drop` |
| `DROP_RATE` / `DROP_SCHEDULE` | `0.2` / `linear` | Drop 超参 |
| `SPARSE_ID` / `SPARSE_VIEW_NUM` | `4` / `9` | hull / 训练视角数 |
| `SKIP_PATH` | `1` | 跳过路径视频步骤 4、10（不影响评测） |
| `LORA_RANK` | `32` | 省显存 |
| `HF_ENDPOINT` | （可选） | LoRA 拉 CLIP 时建议 `https://hf-mirror.com` |

粗训/评测默认：只存最后一轮 ply；render 用 `--not_generate_video --not_saveimages`。  
LOO stage1 仍保留 `chkpnt6000.pth`（stage2 必需）。

路径由 `_common.sh` 相对定位仓库根（含 `3DGS/` 与 `VQA/`）；venv 仍在仓库根 `.venv-ddgs-vggt`。

## 跑法

```bash
# 在仓库根 Sparse-3DGS-heritage-VQA/
source .venv-ddgs-vggt/bin/activate
unset OMP_NUM_THREADS
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0 USE_DROP=1 SKIP_PATH=1
# export HF_ENDPOINT=https://hf-mirror.com   # LoRA 需要时

cd 3DGS/GaussianObject/sh/lineB_gaussianobject
bash 0_run_all.sh            # 全流程
# bash 0_run_all.sh 5        # 粗训已完成时从 LOO 起
# bash run_iter_sweep_classic_vs_vggt.sh       # P3：PSNR–iter 3k/5k/7k/10k
# bash run_depthft_mip360_repair_cadr.sh       # P3：depthft→RV51±CADR
# bash run_garden_classic_vs_depthft_coarse.sh # P3：garden classic vs depthft
# bash run_classic_rv51_repair.sh              # P2：classic粗训→RV51 repair
# bash run_garden_classic_vs_vggt_coarse.sh    # P2：garden classic vs VGGT 粗训
```

LoRA/repair 需 `3DGS/GaussianObject/models/`：**主线** `Realistic_Vision_V5.1.safetensors` + ControlNet-Tile（见 [`models/README.md`](../models/README.md)），以及完整 CLIP。默认 `LORA_EXP`/`REPAIR_TAG` 为 `*_rv51`。

指标 CSV：[`3DGS/eval_results/`](../../eval_results/)。
