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
| **A · 稀疏 3DGS 重建** | VGGT 补点 + GaussianObject + DropGaussian + LOO/LoRA/repair；**CADR** + 写实底模对照 | **主线可跑**（CADR / 底模已落地） |
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
稀疏图 + 位姿 → VGGT visual hull → train_gs（± Drop）→（可选）LOO/LoRA/repair（± CADR）→ results.json
```

- 代码：[`3DGS/GaussianObject/`](3DGS/GaussianObject/)、[`3DGS/vggt/`](3DGS/vggt/)
- 脚本：[`3DGS/GaussianObject/sh/lineB_gaussianobject/`](3DGS/GaussianObject/sh/lineB_gaussianobject/)（[`sh/README.md`](3DGS/GaussianObject/sh/README.md)）
- Drop 参考 submodule：[`3DGS/DropGaussian_release/`](3DGS/DropGaussian_release/)（训练已 plug-in 进 GO renderer）
- **指标归档**：[`3DGS/eval_results/`](3DGS/eval_results/)（CSV；含 LoRA repair / CADR / 底模对照；VGGT depthft 负结果见 [`README_vggt_depthft.md`](3DGS/eval_results/README_vggt_depthft.md)）

### 本仓方法 · CADR（置信度感知扩散修复）

步骤 8 对扩散修复监督乘以 \(w_{\text{need}}\times w_{\text{trust}}\)（视角需求 × CLIP 身份信任），抑制不可靠伪标签灌进 3DGS。

- **设计目标**：相对 RV51 repair（无 CADR），test **PSNR +0.2**（SSIM 不降、LPIPS 不升）。
- **当前实测（kitchen）**：与 `kitchen_drop_rv51` **持平**（ΔPSNR ≈ 0；设计目标未达）。数字见 [`3DGS/eval_results/`](3DGS/eval_results/)。
- 开关：`USE_CONF_WEIGHT=1` → tag `*_rv51_conf`；实现：[`gaussian_object_system.py`](3DGS/GaussianObject/threestudio/systems/gaussian_object_system.py)。

### 本仓尝试 · VGGT depth_head 域适应（负结果，2026-10-15）

冻 `aggregator`，只训 VGGT `depth_head`（COLMAP 稀疏深度 + 每图 `median_scale_l1`）；hull 门控仍为 **GATE_ZOE**。设计目标：相对 G1 粗训 test **PSNR ≥ +0.15**。

两类微调后，在 **mip360/kitchen** 上各自重跑 hull → 粗训 Drop → `ours_10000`（不覆盖 G1）：

| 微调数据 | 权重（保留） | 粗训 GS_DIR（保留） | test PSNR | ΔPSNR vs G1 |
|----------|--------------|---------------------|----------:|------------:|
| G1 原权重 | `3DGS/vggt/models/model.pt` | `kitchen_drop` | **26.141** | — |
| kitchen-only | `depth_head_ft_kitchen.pt` | `kitchen_drop_depthft` | 26.088 | **−0.053** |
| kitchen+bonsai+garden（三场景） | `depth_head_ft_mip360.pt` | `kitchen_drop_depthft_mip360` | 25.978 | **−0.163** |

尺度诊断（kitchen 40 帧）：base / depthft_mip360 的 aligned_L1 ≈ **0.051 / 0.052**（几乎不变）→ 微调几乎未改善 kitchen 几何。hold-out（bonsai+garden 训、kitchen 测）**已跳过**。

**结论：** 成功标准未达；**不作为论文主贡献**。代码、微调权重、粗训产物与 Δ 表均保留（复现/附录用）。详情：[`3DGS/eval_results/README_vggt_depthft.md`](3DGS/eval_results/README_vggt_depthft.md)、[`lineB_kitchen_vggt_depthft.csv`](3DGS/eval_results/lineB_kitchen_vggt_depthft.csv)；入口脚本 `sh/lineB_gaussianobject/*depthft*`、`tools/finetune_depth_head_colmap.py`。设计稿：[`docs/PLAN_vggt_dpt_depth_adapt.md`](docs/PLAN_vggt_dpt_depth_adapt.md)（已归档状态）。

### 本仓贡献 · SD1.5 族写实底模接入

在 GaussianObject + ControlNet-Tile 管线中接入可切换 SD 底模（**须与 LoRA 同底模重训**）。换底模实测 ≈ 持平；论文主对照仍用 RV51 ± CADR，abs16 作附录。

| 本地路径 | 角色 |
|----------|------|
| [`3DGS/GaussianObject/models/Realistic_Vision_V5.1.safetensors`](3DGS/GaussianObject/models/Realistic_Vision_V5.1.safetensors) | 主对照 SD 底模（RV5.1） |
| [`3DGS/GaussianObject/models/absolutereality_v16.safetensors`](3DGS/GaussianObject/models/absolutereality_v16.safetensors) | 附录 SD 底模（AbsoluteReality v1.6） |
| [`3DGS/GaussianObject/models/control_v11f1e_sd15_tile.pth`](3DGS/GaussianObject/models/control_v11f1e_sd15_tile.pth) | ControlNet-Tile 权重（固定配套，勿删） |
| [`3DGS/GaussianObject/models/control_v11f1e_sd15_tile.yaml`](3DGS/GaussianObject/models/control_v11f1e_sd15_tile.yaml) | ControlNet-Tile 配置 |

下载与切底模说明：[`3DGS/GaussianObject/models/README.md`](3DGS/GaussianObject/models/README.md)。

```bash
source .venv-ddgs-vggt/bin/activate
export CUDA_VISIBLE_DEVICES=0 USE_DROP=1 SKIP_PATH=1
# export HF_ENDPOINT=https://hf-mirror.com   # LoRA 拉 CLIP 时
cd 3DGS/GaussianObject/sh/lineB_gaussianobject && bash 0_run_all.sh
# 论文主对照 CADR（复用 *_rv51 LoRA，从步骤 8）：
# SD_CKPT=./models/Realistic_Vision_V5.1.safetensors \
# LORA_EXP=controlnet_finetune/kitchen_drop_rv51 BASE_TAG=kitchen_drop_rv51 \
# bash 0_run_all.sh 8
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

- [x] **VGGT depth_head 域适应**：已跑 kitchen-only / mip360 三场景微调 + 粗训；**负结果已归档**（mip360 ΔPSNR **−0.163**）；不作为主线。见上节与 [`3DGS/eval_results/README_vggt_depthft.md`](3DGS/eval_results/README_vggt_depthft.md)。
- [ ] **CADR 消融 A2/A3**（可选 A4/A5）：同 RV51 LoRA，只改步骤 8（仅 \(w_{\text{trust}}\) / 完整 CADR / 覆盖项 / 自适应 denoise）。
- [ ] **论文主表 C0–C3 核对**：CSV 中 coarse / SD1.5 repair / RV51 / CADR 与论文表述对齐。
- [ ] **（可选）abs16 无 CADR（C4）**：隔离「换底模」与 CADR。
- [ ] **VGGT vs classic hull** / **Drop 消融**：见 [`3DGS/eval_results/`](3DGS/eval_results/) CSV `delta_pending`。
- [ ] **步数 sweep（论文图用）**：同一 VGGT+Drop+RV51 设定下，扫 LoRA `max_steps`（如 1200 / 1800 / 2400 / 3000）与 repair `max_steps`（如 3000 / 4000 / 6000），每档存 ply + test 指标再画曲线；**没跑完不要写「某某区间最好」**。
