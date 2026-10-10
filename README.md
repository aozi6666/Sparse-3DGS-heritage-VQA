# Sparse-3DGS-heritage-VQA

> 论文题目：**受限采集下的 3DGS 三维重建方法以及在文化遗产数字展示中的应用研究**  
> 全文设计与流程图见 [`docs/thesis.md`](docs/thesis.md)。

## 论文目标（摘要）

在受限采集条件下完成稀疏视角 3DGS 重建，得到可自由观看的 3D 资产；再构建面向游客新轨迹的 **空间感知（3D 资产）+ 问答交互 / 展示** 系统：

- 拍不全的文化遗产 → 可漫游的 3DGS 数字资产  
- 现场拍摄时：既知道「看到了什么」，也知道「在 3D 资产里看哪里」  
- 想了解 → AI 讲解/问答；想细看 → 进入 3DGS 自由换视角  

三路任务：**Mage 看懂** → **定位对上** → **3DGS 呈现**，再融合为可问答、可导览的系统。

| 工作 | 内容 | 本仓状态 |
|------|------|----------|
| **A · 稀疏 3DGS 重建** | VGGT 补点 + GaussianObject + DropGaussian +（可选）LOO/LoRA/repair | **主线可跑** |
| **B · 视频问答 / 评测** | Mage-VL 理解 + VLMEvalKit（Video-MME 等） | **代码与指标已接入** [`VQA/`](VQA/) |

更多：[`docs/reproduction/README.md`](docs/reproduction/README.md)（3DGS 复现日记）、[`docs/system/README.md`](docs/system/README.md)（系统对接）。全文话术与流程图亦见仓外 [`毕设.md`](../毕设.md) / [`docs/thesis.md`](docs/thesis.md)。

---

## 核心工作意义

**一句话：** 受限采集建成遗产 3D 资产；游客边拍边问、也能进 3D 细看；定位把实时视角挂回数字世界（不是「对着 3D 模型聊天」）。

1. **离线（方法）** — 稀疏图 / 视频抽帧 → 建成 3DGS 资产（拍不全也能建）。  
2. **在线两条并行**  
   - **拍着看** → Mage 流式问答 / 自动讲（亮实时视频理解）  
   - **点开 3D** → 自由换视角细看（亮数字资产能用）  
3. **定位的用处** — 把「手机现在拍的这块」对上「3D 里的哪一块」。  
   - 无定位：问答只懂画面，和 3D 遗产对不上号  
   - 有定位：问答知道你在看哪一处，并可切到 3D 对应视角  

**定位 = 焊条：** 让流式视频问答与已重建的 3D 文化遗产资产发生空间关联。

---

## 工作 A：稀疏视角 3DGS 重建

```text
稀疏图 + 位姿 → VGGT visual hull → train_gs（± Drop）→（可选）LOO/LoRA/repair → results.json
```

- 代码：[`3DGS/GaussianObject/`](3DGS/GaussianObject/)、[`3DGS/vggt/`](3DGS/vggt/)
- 脚本：[`3DGS/GaussianObject/sh/lineB_gaussianobject/`](3DGS/GaussianObject/sh/lineB_gaussianobject/)（[`sh/README.md`](3DGS/GaussianObject/sh/README.md)）
- Drop 参考 submodule：[`3DGS/DropGaussian_release/`](3DGS/DropGaussian_release/)（训练已 plug-in 进 GO renderer）
- **指标归档**：[`3DGS/eval_results/`](3DGS/eval_results/)（CSV；含 LoRA repair 增益，VGGT/Drop 对照待补）

```bash
source .venv-ddgs-vggt/bin/activate
export CUDA_VISIBLE_DEVICES=0 USE_DROP=1 SKIP_PATH=1
# export HF_ENDPOINT=https://hf-mirror.com   # LoRA 拉 CLIP 时
cd 3DGS/GaussianObject/sh/lineB_gaussianobject && bash 0_run_all.sh
```

权重/数据不进 Git（VGGT、SD/ControlNet、CLIP、mip360 kitchen 等），见复现文档。

---

## 工作 B：Mage-VL + VLMEvalKit

| 路径 | 说明 |
|------|------|
| [`VQA/mage_vl/`](VQA/mage_vl/) | 基于 microsoft/Mage 的 `mage_vl` + streaming 增量；**无** `data/`/`models/` |
| [`VQA/VLMEvalKit/`](VQA/VLMEvalKit/) | 评测框架 **vendored 完整代码（含本仓改动）** |
| [`VQA/patches/vlmevalkit/`](VQA/patches/vlmevalkit/) | 相对上游 open-compass 的 diff（说明用） |
| [`VQA/eval_results/`](VQA/eval_results/) | **Video-MME 等准确率结果**（必留） |

`mage_flow`（文生图）**不在本仓**，仅存于机器上仓外 `../Mage/mage_flow` 等。  
权重与数据集放仓外或本地 ignore 目录；环境见下表与 [`VQA/README.md`](VQA/README.md)。

---

## 环境（uv，目录可不进库）

| 环境（本地） | 用途 | freeze |
|--------------|------|--------|
| `.venv-ddgs-vggt` | 工作 A | [`envs/requirements-venv-ddgs-vggt.txt`](envs/requirements-venv-ddgs-vggt.txt) |
| `ddgs` | 旧 DDGS 参考 | [`envs/requirements-ddgs.txt`](envs/requirements-ddgs.txt) |
| `../VLMEvalKit/vlmeval_env` 或自建 | 工作 B 评测 | [`envs/requirements-vlmevalkit.txt`](envs/requirements-vlmevalkit.txt) |
| `../Mage_ViT/mage` 或自建 | 工作 B Mage | [`envs/requirements-mage-vit.txt`](envs/requirements-mage-vit.txt) |

```bash
git submodule update --init --recursive   # DropGaussian_release 等
# 评测时将 editable 路径改为本仓 VQA/VLMEvalKit
```

---

## 布局

见 [`STRUCTURE.txt`](STRUCTURE.txt)。忽略：`.venv*`、`ddgs/`、`_archive/`、`**/data/`、`**/output/` 大产物、权重。

---

## TODO（后续慢慢做）

- [ ] **步数 sweep（论文图用）**：同一 VGGT+Drop+RV51 设定下，扫 LoRA `max_steps`（如 1200 / 1800 / 2400 / 3000）与 repair `max_steps`（如 3000 / 4000 / 6000），每档存 ply + test 指标再画曲线；**没跑完不要写「某某区间最好」**。主线默认已是 RV5.1（`*_rv51`）。
