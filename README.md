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
<<<<<<< HEAD
| **A · 稀疏 3DGS 重建** | init（classic / VGGT±depthft）→ Drop 粗训 → LoRA repair（± CADR / 底模） | **消融已齐**（见下节结论） |
| **B · 视频问答 / 评测** | Mage-VL 理解 + VLMEvalKit（Video-MME 等） | **代码与指标已接入** [`VQA/`](VQA/) |
=======
| **A · 稀疏 3DGS 重建** | VGGT 补点 + GaussianObject + DropGaussian + LOO/LoRA/repair；**CADR** + 写实底模对照 | **主线可跑**（CADR / 底模已落地） |
| **B · 视频问答 / 评测** | Mage Frames-32＋连续字幕2048；Query-IP历史路线保留 | **冻结test225已完成**：163/225；详见最新结果总结 |
>>>>>>> vlm

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
稀疏图 + 位姿
  → ① init 点云（classic hull / VGGT-enhanced / ± depth_head 微调）
  → ② 粗训 3DGS（± DropGaussian）
  → ③ LOO + ControlNet-Tile LoRA
  → ④ Gaussian repair（± CADR；可换 SD1.5 写实底模）
  → test results.json → 3DGS/eval_results/*.csv
```

| 路径 | 说明 |
|------|------|
| 代码 | [`3DGS/GaussianObject/`](3DGS/GaussianObject/) · [`3DGS/vggt/`](3DGS/vggt/) |
| 编排 | [`sh/lineB_gaussianobject/`](3DGS/GaussianObject/sh/lineB_gaussianobject/)（[`sh/README.md`](3DGS/GaussianObject/sh/README.md)） |
| 指标 | [`3DGS/eval_results/`](3DGS/eval_results/)（CSV + SUMMARY） |

### 主模块：定位 · 作用 · 收益 · 意义

| 模块 | 管线位置 | 作用 | 实测收益（test PSNR） | 论文意义 |
|------|----------|------|----------------------|----------|
| **① Classic visual hull** | 步骤 1 默认强基线 | COLMAP/掩码几何围成初始点云 | kitchen≈**27.1**；garden≈**33.5** | **粗训主收益在 init**；遗产受限采集下优先保几何密度与紧致 bbox |
| **①′ VGGT-enhanced hull** | 同级对照（原版 `model.pt`） | 用 VGGT 深度质量评估增强 hull | kitchen 相对 classic **≈−1.0 dB**；garden 仅 **≈−0.11** | VGGT 是可选增强而非粗训 SOTA；**场景依赖**，勿写成全局 −1 dB |
| **①″ Depth-adapt** | VGGT depth_head 任务微调后再 enhanced | 域适应深度先验（mip360 ckpt） | kitchen 粗训 **−0.16**；garden **+0.16 vs 原版 VGGT**；接 repair 后 kitchen **≥ 原版 A0** | 可写「任务微调」；粗训不稳定，**不宜宣称普适大涨点** |
| **② DropGaussian** | 粗训正则 | 训练中随机丢高斯抑过拟合 | classic 上≈0；VGGT 上约 **+0.09** | 弱/条件性增益；配合差 init 略有用，不是主贡献 |
| **③ LoRA repair** | LOO 对 + Tile LoRA → 步骤 8–9 | 扩散先验修稀视图伪影 | VGGT 粗训→RV51：**+0.80**（相对 26.14）；弱 depthft 粗训→RV51：**+1.04** | **下游主涨点**；修复补的是渲染伪影，**抹不平 init 几何差距** |
| **④ CADR** | repair 时 \(w_{need}\times w_{trust}\) | 置信度门控修复强度 | vs 无 CADR：**≈0～+0.05**（目标 +0.2 **未达**）；A5 略伤默认关 | 方法完整可报；实证接近噪声，主文勿夸大 |
| **④′ 写实底模** | RV51 / abs16 换 SD1.5 | 更强纹理先验（须与 LoRA 同底重训） | RV51 vs SD1.5 ≈**+0.10**；abs16 无实质涨点 | 工程可切换；**不是指标主杠杆** |

**负对照：** 纯 VGGT 点云（无 classic hull）+ depthft → **11.78**（≈−14 dB）→ 必须保留 visual-hull 几何骨架（[`PLAN_vggt_pure_depthft.md`](docs/PLAN_vggt_pure_depthft.md)）。

**欠训排查：** PSNR–iter（3k→10k）全程 classic ≫ VGGT，10k 仍 ≈**−1.5 dB** → kitchen 的 init 差距**不是欠训假象**（[`lineB_kitchen_iter_sweep.csv`](3DGS/eval_results/lineB_kitchen_iter_sweep.csv)）。

### 关键实验数字（mip360 / test）

**粗训 2×2（kitchen）** · [`SUMMARY_coarse_2x2.md`](3DGS/eval_results/SUMMARY_coarse_2x2.md)

| | 无 Drop | Drop |
|--|--------:|-----:|
| classic | **27.11** | 27.10 |
| VGGT-enhanced | 26.05 | 26.14 |

ply：classic 8616 点、紧 bbox；VGGT-enhanced 2874 点、更散 → 与 −1 dB 同向。

**多场景粗训（Drop=1）**

| 场景 | classic | VGGT / depthft | 解读 |
|------|--------:|---------------:|------|
| kitchen | 27.10 | 26.14 / 25.98 | init 主导；depthft 粗训略伤 |
| garden | 33.49 | 33.37 / **33.53** | 差距缩小；depthft 微胜 classic |

**Repair 主表（C0–C4 + P2/P3）** · [`SUMMARY_repair_cadr.md`](3DGS/eval_results/SUMMARY_repair_cadr.md) · [`SUMMARY_depthft_expand.md`](3DGS/eval_results/SUMMARY_depthft_expand.md)

| ID / 设定 | test PSNR | 要点 |
|-----------|----------:|------|
| C0 classic 粗训 | ≈**27.11** | 粗训最优 |
| C1 SD1.5 repair（VGGT init） | ≈26.94 | vs 旧 VGGT 粗训 +0.80 |
| C2/A0 RV51 无 CADR | **26.97** | 底模边际 |
| C3/A3 RV51+CADR | **26.96** | vs C2 持平 |
| C4 abs16 无 CADR | **26.93** | 换底模无实质收益 |
| classic→RV51（P2） | **27.09** | vs A0 +0.12；vs 自身粗训持平 |
| depthft→RV51 / +CADR（P3） | **27.02 / 27.07** | 拉回弱粗训；略超原版 A0/A3 |

### 一句话结论（工作 A）

1. **主杠杆 = 初始点云几何（classic hull）+ LoRA repair**；Drop / CADR / 换底模 / depthft 均为弱或场景条件性。  
2. **VGGT 与 depth-adapt** 定位为可选深度先验与任务微调，**不是**粗训主 SOTA；跨场景幅度差一个数量级。  
3. **Repair 不能翻盘 init**：classic 优势会传到下游，但在已足够好的 classic 粗训上 repair 几乎不涨。  
4. 遗产导览侧：优先保证 **可漫游、几何稳的 3DGS 资产**（C0 级），再用 repair 抬纹理；CADR/底模作方法完整度与附录。

### 跑通与权重

```bash
source .venv-ddgs-vggt/bin/activate
export CUDA_VISIBLE_DEVICES=0 USE_DROP=1 SKIP_PATH=1
cd 3DGS/GaussianObject/sh/lineB_gaussianobject && bash 0_run_all.sh
# CADR 主对照（复用 RV51 LoRA，从步骤 8）：
# SD_CKPT=./models/Realistic_Vision_V5.1.safetensors \
# LORA_EXP=controlnet_finetune/kitchen_drop_rv51 BASE_TAG=kitchen_drop_rv51 \
# bash 0_run_all.sh 8
```

| 本地权重 | 角色 |
|----------|------|
| [`models/Realistic_Vision_V5.1.safetensors`](3DGS/GaussianObject/models/Realistic_Vision_V5.1.safetensors) | 主对照 SD 底模 |
| [`models/absolutereality_v16.safetensors`](3DGS/GaussianObject/models/absolutereality_v16.safetensors) | 附录底模（C4） |
| [`models/control_v11f1e_sd15_tile.pth`](3DGS/GaussianObject/models/control_v11f1e_sd15_tile.pth) (+ `.yaml`) | ControlNet-Tile |
| [`3DGS/vggt/models/model.pt`](3DGS/vggt/models/model.pt) | 原版 VGGT |
| [`depth_head_ft_mip360.pt`](3DGS/vggt/models/depth_head_ft_mip360.pt) | depth-adapt（任务微调） |

详见 [`models/README.md`](3DGS/GaussianObject/models/README.md)。权重/大数据不进 Git。

---

<<<<<<< HEAD

## 工作 B：Mage-VL + VLMEvalKit

| 路径 | 说明 |
|------|------|
| [`VQA/mage_vl/`](VQA/mage_vl/) | 基于 microsoft/Mage 的 `mage_vl` + streaming 增量；**无** `data/`/`models/` |
| [`VQA/VLMEvalKit/`](VQA/VLMEvalKit/) | 评测框架 **vendored 完整代码（含本仓改动）** |
| [`VQA/patches/vlmevalkit/`](VQA/patches/vlmevalkit/) | 相对上游 open-compass 的 diff（说明用） |
| [`VQA/eval_results/`](VQA/eval_results/) | **Video-MME 等准确率结果**（必留） |

`mage_flow`（文生图）**不在本仓**，仅存于机器上仓外 `../Mage/mage_flow` 等。  
权重与数据集放仓外或本地 ignore 目录；环境见下表与 [`VQA/README.md`](VQA/README.md)。
=======
## 工作 B：Mage字幕增强（当前）

## 当前结论（2026-11-03，字幕路线已完成冻结评测）

保留Frames-32＋连续补句＋时间标记，固定2048字幕token；无训练、无Agent/API。test225三路正确数：无字幕140、原对齐154、连续163（72.44%）。相对无字幕净增23题／10.22个百分点，聚类95%区间[5.33,15.11]；相对原对齐净增9题／4个百分点，区间[-0.44,8.46]，尚不能证明补句独立显著优于原对齐。

short219/dev180/val90/test225连续字幕分别175/119/54/163题正确，四阶段互斥覆盖本地714题／238视频，不能混称714题独立测试或官方完整基准。原对齐val90负结果、3072预算负结果及API pilot负结果全部保留。当前不再调字幕或重复跑本地full；Agent暂停，文旅闭环和重建对问答的贡献尚待验证。

完整三路数据、分时长统计、5组图、成本和10个案例见[最新最终结果总结](docs/SUMMARY_mage_subtitle_final_results_20261103.md)。历史内容若与本节不符，以本节及最新总结为准；历史实验不删除。


### Query-IP历史路线（独立协议，不能与字幕结果直接混比）

**应用场景（动机）：** 游客随拍、视角有限的景点问答——同一段短视频，问牌匾 vs 问屋檐，需要的视觉证据不同；Codec 按运动/残差选 Patch 未必对准问题；证据不足时应说明不确定（拒答需另做）。  
**核心研究问题：** 有限视角下、**固定视觉 Token 预算**的**问题相关证据选择**（不是泛旅游聊天，也不是单纯压码率）。  
**实验结论（2026-10-27 保留）：**  
- **主线 A（可写）：** 同 K **Codec 融合** `queryip` **63.5%** ≈ +4 pp vs random（视频 · Mage 已留 patch）。  
- **按问题选区未做成：** CLIP / αβ / OCR 硬切 textseat（37.3%=shuf，vs full −46.8）→ **失败消融，不作贡献**。  
- **加戏 B：** 软门控、持平即成功（可选）。换方案时以上结论保留。  

详情：[`VQA/README.md`](VQA/README.md) · [`docs/PLAN_mage_query_IP_select.md`](docs/PLAN_mage_query_IP_select.md)。

| 同 K · short200 · kr=0.75 | Acc | 备注 |
|------|-----|------|
| **queryip · 主线 A** | **63.5%** | [`clip_frame_n200/`](VQA/eval_results/queryip_samek/clip_frame_n200/) |
| random | 59.5% | 同上 |
| CLIP / region·shuf / edge·shuf | 58.5 / 58.5·60.5 / 60.0·62.0 | 失败 |
| OCR textseat / full（v1 proxy） | 37.3 / 84.1 | [`ocrbench_v1_image_textseat/`](VQA/eval_results/queryip_samek/ocrbench_v1_image_textseat/) |

| 路径 | 说明 |
|------|------|
| [`VQA/README.md`](VQA/README.md) | **应用场景全文** + 实验结论 |
| [`VQA/Explore-Then-Select/`](VQA/Explore-Then-Select/) | Query-IP + [`COMMANDS.md`](VQA/Explore-Then-Select/COMMANDS.md) |
| [`VQA/eval_results/`](VQA/eval_results/) | 指标归档 |
| [`docs/PLAN_mage_query_IP_select.md`](docs/PLAN_mage_query_IP_select.md) | 结论 · 证据 · 后续实验 |

`mage_flow` **不在本仓**。权重 / 大数据集放仓外或 ignore。
>>>>>>> vlm

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

## TODO

**工作 A 消融已齐**（2×2 / V2 负对照 / CADR A-full+C4 / P2 classic-repair+garden / P3 iter-sweep+depthft）。详表：[`3DGS/eval_results/`](3DGS/eval_results/)。

**后续可选**

<<<<<<< HEAD
- [ ] 用 [`lineB_kitchen_iter_sweep.csv`](3DGS/eval_results/lineB_kitchen_iter_sweep.csv) 出论文 **PSNR–iter** 图
- [ ] **V2 尺度对齐**（非必须；11.78 负对照即可）
- [ ] 工作 B：Cultural LoRA / codec 选区等（见 [`docs/PLAN_mage_cultural_lora_location.md`](docs/PLAN_mage_cultural_lora_location.md)）
=======
### 工作 B · Query-IP / Mage（下列旧TODO为历史路线；当前字幕状态见上文）

- [x] **同 K 主表 / 复现**：保留 `queryip`；vs random 点估计约 **+4 pp**（63.5 vs 59.5 @`clip_frame_n200`；主表曾 64.0）
- [x] **失败消融归档**：bitcost；CLIP 帧 E4；区域 v1；**边界 edge**；方案 B **不做**
- [x] **E2 α/β 三角**：`queryip`≡dyn；sta ±1 题 → **题型自适应不作贡献**
- [x] **行 A / 行 B 旁证** + **proxy K 扫描**（0.3–0.75 赢；0.9 不赢）
- [x] **停 QueryIP 语义选区**（CLIP 重排；非否定一切语义）
- [x] **OCRBench v1 硬切 textseat 负结果归档**；Mage baseline Norm 83.4
- [x] **定案主线 A / 加戏 B**（换方案前结论保留）
- [ ] **【优先】主线 A 收口**：McNemar · 失败表 · Acc–K
- [ ] （可选）加戏 B 软门控冒烟 · 效率曲线 · 流式 / Cultural LoRA
- [ ] ~~同 K×714~~ / ~~CLIP 选区~~ / ~~αβ~~ / ~~OCR 硬切打 Mage~~ → **不做**
>>>>>>> vlm
