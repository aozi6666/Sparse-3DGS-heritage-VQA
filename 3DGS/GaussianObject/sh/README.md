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
# bash 2b_train_gs_drop_ablation.sh
```

LoRA/repair 需 `3DGS/GaussianObject/models/` 权重与完整 CLIP（非空 LFS 指针）。

指标 CSV（PSNR/SSIM/LPIPS 与增益）：仓库 [`3DGS/eval_results/`](../../eval_results/)。
