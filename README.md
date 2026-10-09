# 文化遗产数字展示系统（系统仓 · 第一阶段）

本仓库是毕设系统的 **GitHub 主仓（第一阶段）**，规划两大工作：

| 工作 | 内容 | 本仓状态 |
|------|------|----------|
| **A · 稀疏 3DGS 重建** | VGGT 补点 + GaussianObject 粗训 + DropGaussian 正则 +（可选）LOO/LoRA/repair | **现行主线，可跑** |
| **B · 视频问答 / 导览** | Mage 理解 + 评测（VLMEvalKit）等，与 3D 资产空间关联 | **占位**，见 [`VQA/`](VQA/) |

详细复现日记（环境、旧实验记录）见 [`docs/reproduction/README.md`](docs/reproduction/README.md)。  
系统如何对接重建与问答，见 [`docs/system/README.md`](docs/system/README.md)。

---

## 工作 A：稀疏视角 3DGS 重建

```text
稀疏图 + 位姿
  → VGGT visual hull 点云
  → GaussianObject train_gs（± DropGaussian）
  →（可选）LOO → LoRA → repair
  → 新视角渲染 / results.json
```

- 代码：[`GaussianObject/`](GaussianObject/)（含 Drop 接入的 `gaussian_renderer` / `train_gs.py`）
- 几何：[`vggt/`](vggt/)（权重 `vggt/models/model.pt` 需自行下载，不进库）
- 脚本：[`GaussianObject/sh/lineB_gaussianobject/`](GaussianObject/sh/lineB_gaussianobject/)（说明见 [`sh/README.md`](GaussianObject/sh/README.md)）
- Drop 参考实现（submodule）：[`DropGaussian_release/`](DropGaussian_release/) ← [DCVL-3D/DropGaussian_release](https://github.com/DCVL-3D/DropGaussian_release)  
  训练时用的是已 plug-in 进 GO 的 opacity dropout，不必另跑该仓。

### 一条命令（主线）

```bash
# 依赖：uv 环境、数据、VGGT/SD/ControlNet/CLIP 权重（见 docs/reproduction）
source .venv-ddgs-vggt/bin/activate
export CUDA_VISIBLE_DEVICES=0 USE_DROP=1 SKIP_PATH=1
# 国内 LoRA 需 CLIP：export HF_ENDPOINT=https://hf-mirror.com

cd GaussianObject/sh/lineB_gaussianobject
bash 0_run_all.sh          # 全流程
# bash 0_run_all.sh 5      # 粗训已完成时从 LOO 起
# bash 2b_train_gs_drop_ablation.sh   # ±Drop 粗训对照
```

### 权重与数据（不进 Git）

| 资源 | 位置（本地） | 说明 |
|------|----------------|------|
| VGGT | `vggt/models/model.pt` | 官方 VGGT 权重 |
| SD 1.5 / ControlNet tile | `GaussianObject/models/` | LoRA/repair 用 |
| CLIP ViT-L/14 | HF 或完整本地目录 | LoRA 文本编码；勿用空 LFS 指针 |
| MipNeRF360 kitchen 等 | `GaussianObject/data/` | 含 COLMAP / `sparse_*.txt` |

---

## 工作 B：问答（占位）

代码暂在仓库外（同级目录 `../Mage`、`../VLMEvalKit`），本仓用 [`VQA/README.md`](VQA/README.md) 预留接入位置。  
环境 freeze 已放在 [`envs/`](envs/)，便于后续迁入后复现。

---

## 环境与依赖复现（uv）

用 [`uv`](https://github.com/astral-sh/uv) 管理；**不要提交** `.venv*` / `ddgs/` 目录本身，只提交 `envs/` 快照。

| 环境目录（本地） | Python / Torch | 用途 | freeze |
|------------------|----------------|------|--------|
| `.venv-ddgs-vggt` | 3.10 / 2.5.1+cu124 | **主线**：GO + VGGT + Drop | [`envs/requirements-venv-ddgs-vggt.txt`](envs/requirements-venv-ddgs-vggt.txt) |
| `ddgs` | 3.10 / 2.11+cu128 | 旧 DDGS lineA（归档参考） | [`envs/requirements-ddgs.txt`](envs/requirements-ddgs.txt) |
| `vlmeval_env`（仓外） | 3.11 / 2.11+cu128 | **工作 B**：VLMEvalKit 评测 | [`envs/requirements-vlmevalkit.txt`](envs/requirements-vlmevalkit.txt) |
| `mage`（仓外） | 3.11 / 2.11+cu128 | **工作 B**：Mage_ViT | [`envs/requirements-mage-vit.txt`](envs/requirements-mage-vit.txt) |

```bash
# 主线环境
uv venv .venv-ddgs-vggt --python 3.10
uv pip install -r envs/requirements-venv-ddgs-vggt.txt --python .venv-ddgs-vggt/bin/python
# CUDA 扩展需在 GaussianObject 下 --no-build-isolation；transformers 建议钉 4.44.2（兼容 torch 2.5）

git submodule update --init --recursive   # 含 DropGaussian_release 与 GO submodules
```

注意：`requirements-vlmevalkit.txt` 中的 `-e ../VLMEvalKit` 指向仓外路径，工作 B 迁入前请按本机布局调整。

---

## 仓库布局

见 [`STRUCTURE.txt`](STRUCTURE.txt)。本地存在但默认不进库：`.venv*`、`ddgs/`、`_archive/`、`**/data/`、`**/output/`、大权重。
