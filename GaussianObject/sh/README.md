# GaussianObject 训练脚本

路径：`/root/autodl-tmp/3DGS`。现行主线：`lineB_gaussianobject/`（默认 `USE_DROP=1`）。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `USE_DROP` | `1` | DropGaussian；目录用 `kitchen_drop` |
| `DROP_RATE` / `DROP_SCHEDULE` | `0.2` / `linear` | Drop 超参 |
| `SPARSE_ID` / `SPARSE_VIEW_NUM` | `4` / `9` | hull / 训练视角数 |
| `SKIP_PATH` | `1` | 跳过路径视频步骤 4、10（不影响评测） |
| `LORA_RANK` | `32` | 省显存（原脚本 64） |
| `START`（`0_run_all.sh` 参数） | `1` | 从第 N 步起跑 |

粗训/评测默认：只存最后一轮 ply；render 用 `--not_generate_video --not_saveimages`；训完删 tfevents/`input.ply`。LOO 仍保留 `chkpnt6000.pth`（stage2 必需）。

## 推荐：粗训已完成（kitchen_drop）→ 从 LOO 跑整套

```bash
source /root/autodl-tmp/3DGS/.venv-ddgs-vggt/bin/activate
unset OMP_NUM_THREADS
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export USE_DROP=1
export SKIP_PATH=1

cd /root/autodl-tmp/3DGS/GaussianObject/sh/lineB_gaussianobject
bash 0_run_all.sh 5
```

全流程（含 hull+粗训）：`bash 0_run_all.sh`  
仅 ±Drop 粗训对照：`bash 2b_train_gs_drop_ablation.sh`

LoRA/repair 需 `GaussianObject/requirements.txt` 与 `models/` 权重齐全。
