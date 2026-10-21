# Mage 按问题调 I/P 配额（Query-aware codec 选区）— 实验计划

> **日期：** 2026-10-15（修订：2026-10-21 short-60 同 K 归档）  
> **状态：** 行 B 全量 **62.3%**（非同 K）。路线 B 同 K：soccer `SAMEK_OK`；Video-MME short 前 60 题 **queryip 61.7% > random 58.3% > prefix 56.7%**（弱正信号，n=60）。详见 §5.4。命令见 [`COMMANDS.md`](../VQA/Explore-Then-Select/COMMANDS.md)。  
> **范围：** 仅 `VQA/` 选区前向包装；路线 B 不改 `mage_vl`；路线 A（真分导出）延期且可能需改 Mage/`cv-preinfer`。  
> **顺序：** ① 冒烟 / 行 B 全量（已完成）→ ② soccer 同 K（已完成）→ ③ short-60 同 K（已完成）→ ④（可选）扩 short 全量 / 路线 A → Cultural LoRA。

---

## 0. 锁定方案（必读 · 主创新）

**简单版（推荐，本计划只做这一档）：**

> Mage 已有 **运动向量（MV）与残差（residual）评分**，不必从零开发。  
> 根据**问题类型**，**动态调整**这两路分数的权重，再在固定总 token 预算下做 patch 选留。  
> **不重训** Mage-ViT / LLM；实现难度不大；叙事是「Query × codec 联合评分」，不是两库硬拼接。

| 问题倾向 | 权重倾向 | 效果 |
|----------|----------|------|
| 静态细节（纹饰、题记、材质） | **压低** MV/残差权重（或相对抬高 I/锚帧、低运动区） | 少被 ego-motion / 晃动背景占满预算 |
| 动态事件（人流、仪式、动作） | **抬高** MV/残差权重 | 更保留真正变化的 P patch |

```text
score' = α(q) · score_MV + β(q) · score_residual
         （α、β 由问题类型决定；实现时可用离散两档或简单规则表）
→ 在预算 K 内按 score' 选 patch（I 帧策略与 Mage 默认兼容，见实现注）
```

| 项 | 决定 |
|----|------|
| 主创新 | 上式：**问题类型 → 重加权 Mage 已有 MV/残差**（ETS 的 query-adaptive 思想落在 codec 分数上） |
| QuoTA | **保持可选**（流程图虚线；时间够再加片段级预算） |
| Cultural LoRA | 新选区就绪后、同一前向再训 |
| FlexSelect / 新记忆系统 / 完整 ego-motion 估参 | **不做**（见 §7） |

**实现注（避免踩坑）：** Mage 默认常「I 帧全留」；静态题若只改 `patch_positions` 而不动锚帧/帧间预算，I 侧可能加不上去。落地时优先改 **MV/残差权重与 P 侧 top-k**，必要时再调锚帧密度或 I/P 帧间预算比——仍不重训。

---

## 0.1 Soccer 冒烟结果（2026-10-21 · `SMOKE_OK`）

视频：`VQA/mage_vl/models/examples/soccer-broadcast.mp4`；`keep_ratio=0.75`；32 frame / HEVC。

| 设置 | n_patches | n_vis_tokens≈ | 备注 |
|------|-----------|---------------|------|
| DEFAULT（无重加权） | 18432 | 4608 | 答出白衣球员带球 |
| REWEIGHT static | 13824 | 3456 | α=0.25 β=0.75；答 `ATLANTA` |
| REWEIGHT dynamic | 13824 | 3456 | α=0.85 β=0.15；同 K |

- 共享 `budget_k=13824`；`same_patch_budget=True`
- `different_kept_canvases=True`；`|Δ mean_motion_kept|=0.0201`（dynamic 更高）
- 终端末行：`SMOKE_OK: selection differs by question type under fixed K; answers produced.`

---

## 1. 背景（解决什么问题）与题目关系

题目见 [`thesis.md`](thesis.md)（仓外原文 `/root/autodl-tmp/毕设.md`）：

> 受限采集下的 3DGS 三维重建方法以及在文化遗产数字展示中的应用研究

系统三路：

| 路 | 职责 | 本文件是否覆盖 |
|----|------|----------------|
| 3DGS | 「呈现」：稀疏/拍不全 → 可漫游数字资产 | 否（拍不全/缺视角仍归此路） |
| 定位 | 「对上」：手机视角 ↔ 3D 资产区域 | 否 |
| Mage | 「看懂」：现场视频 → 导览问答 | **是（仅选区）** |

**本文件只补 Mage 看懂侧的缺口：**

- 景区手持拍摄时，「运动」多半是摄像者在动（ego-motion），不是文物在动。
- Mage codec 粗筛保留的是「相对上一帧变了的 patch」，**不等于**「对当前问题有用的 patch」。
- 因此：保留 codec 减 token，再在接缝上按**问题类型重加权 MV/残差**（遗产/静态题压运动分、偏保细节）——见 **§0**。

**明确不声称：** 用选区解决「有些视角拍不到」——缺视角仍靠 3DGS + 定位进 3D。

**边界：** 不做实时；长视频非优先；不做新滑动窗口 / KV / 记忆系统（沿用 Mage 现成窗与 gate 即可）。

---

## 2. 核心思路

> 细节与公式以 **§0** 为准；本节展开流程与依赖关系。

### 2.1 一句话

```text
Mage codec（保留 MV / residual 分数）
  + 按问题类型重加权 α(q)·MV + β(q)·residual，固定预算 K 选 patch  ← 主创新（简单版）
  ± QuoTA 片段级 query 预算（可选）
  ✗ FlexSelect 不做
→ 同一前向再 Cultural LoRA
```

### 2.2 接缝规则（与 §0 同一套）

| 问题倾向 | 对 Mage 已有分数做什么 | 典型问法 |
|----------|------------------------|----------|
| 静态细节（纹饰、题记、材质、形制） | **降低** MV/残差权重（静态偏细节 / 偏 I） | 「这块纹样是什么」「碑上写了什么」 |
| 动态事件（人流、仪式、明显动作） | **提高** MV/残差权重（动态偏 P） | 「刚才谁走过」「仪式怎么进行」 |

- 不重训；ETS 只提供「按问题调 static/dynamic」的思想，**分数来自 Mage codec**，联合加权是本仓接缝。
- 「你走不走」≠「该抬 MV」：走路是采集状态；权重由**问题类型**决定。
- 通用榜不承诺硬涨点；主打 **同预算 Video-MME Acc / 同 Acc 更少 token**。

### 2.3 方法流程图

主路径实线；QuoTA 为可选虚线。

```mermaid
flowchart TD
  Video[现场导览视频] --> Codec[Mage_codec算出_MV与残差分]
  Codec --> QType[问题类型判定]
  QType --> Reweight[按问题重加权_MV残差_主创新]
  Reweight --> Select[固定预算K选patch]
  Select -->|"主路径"| ViT[Mage_ViT加Qwen3解码]
  Select -.->|"可选"| QuoTA[QuoTA片段级query预算]
  QuoTA -.-> ViT
  ViT --> Eval[评测_同预算Acc或同Acc更少token]
  Eval --> LoRA[同一前向_Cultural_LoRA]
```

---

## 3. 依赖的已有工作

| 组件 | 定位 | 作用 / 提供什么 | GitHub / 路径 |
|------|------|-----------------|---------------|
| **Mage-VL codec** | 底座粗筛（保留） | I 全 patch、P 按 MV/residual 稀疏；>75% 减 token；codec+HEVC / 流式接口 | https://github.com/microsoft/Mage ；本地 [`VQA/mage_vl/`](../VQA/mage_vl/) |
| **Explore-Then-Select** | 思想来源（非原样搬代码） | query-adaptive static/dynamic；本仓落成「问题类型 → 重加权 Mage MV/残差」 | https://github.com/ANDgate99/Explore-Then-Select ；完整上游树在仓外 `/root/autodl-tmp/Explore-Then-Select` |
| **QuoTA** | **可选**（流程图虚线，保持可选） | 用户提问时片段/帧级 query 预算 | https://github.com/MAC-AutoML/QuoTA |
| **Cultural LoRA（本仓）** | 域适配（排在选区后） | 冻 `model.visual`，只训 LLM LoRA；MSVD 8k + CiQi 16k | [`VQA/mage_vl/cultural_lora/`](../VQA/mage_vl/cultural_lora/)；计划 [`PLAN_mage_cultural_lora_location.md`](PLAN_mage_cultural_lora_location.md) |
| FlexSelect | **不做** | — | 见 §7 |

**本地接入点（MVP 已落地）：**

- 实现目录：[`VQA/Explore-Then-Select/mage_query_ip/`](../VQA/Explore-Then-Select/mage_query_ip/)（包装 Mage codec 输出；**不改** `mage_vl` 源文件）
- VLMEvalKit 模型名：`Mage-VL`（行 A）/ `Mage-VL-QueryIP`（行 B）
- 命令：[`VQA/Explore-Then-Select/COMMANDS.md`](../VQA/Explore-Then-Select/COMMANDS.md)
- 说明：[`VQA/Explore-Then-Select/README_MAGE_QUERY_IP.md`](../VQA/Explore-Then-Select/README_MAGE_QUERY_IP.md)
- 评测硬约束：对比组 **总 token 预算对齐**；记录真实墙钟耗时 + Avg #vis-token

---

## 4. 依赖的数据集、权重与产物

### 4.1 权重

| 项 | 路径 / 来源 |
|----|-------------|
| Mage-VL 基座 | `VQA/mage_vl/models/`（HF `microsoft/Mage-VL`，`local_files_only`） |
| Cultural LoRA | 仅存 `adapter/`；**须在新选区前向下重训**后才进主表 |

### 4.2 训练数据（LoRA，选区就绪后）

| 数据 | 用量 | 路径 |
|------|------|------|
| MSVD-QA | 8k QA | `VQA/mage_vl/data/MSVD-QA/train_8k/` |
| CiQi-VQA | 16k | `VQA/mage_vl/data/CiQi-VQA/`（`train_16k.jsonl`） |
| 合计磁盘（基座+子集） | ≈ **18.6 G** | 见 Cultural LoRA 计划 |

### 4.3 评测数据

| 集 | 用途 | 状态 |
|----|------|------|
| **Video-MME**（codec+HEVC，32frame） | 主对照；行 A / 行 B Acc + Avg #vis-token | 行 A → [`Video-MME.md`](../VQA/eval_results/Video-MME.md)；行 B → [`Mage-VL-QueryIP_Video-MME.md`](../VQA/eval_results/Mage-VL-QueryIP_Video-MME.md)（**62.3%**） |
| 效率日志 | Avg #视觉 token、相对默认 Mage 的 token 比 | [`Mage-VL-QueryIP_videomme_tokens.jsonl`](../VQA/eval_results/Mage-VL-QueryIP_videomme_tokens.jsonl)（714 行） |

---

## 5. 实验表（Video-MME · 行 A vs 行 B）

**主结论写法：** 同 token 预算下比准确率，或同准确率下比更少 token。  
**不承诺** Video-MME 硬涨 1–2 点。——行 B 实测符合「不涨、略降」。

| 设置 | Video-MME overall | Short | Medium | Long | Avg #vis-token | 相对 token 比 |
|------|-------------------|-------|--------|------|----------------|---------------|
| A. 默认 Mage codec（`Mage-VL`） | **63.7%** | 75.8% | 63.6% | 52.7% | （未单独记 jsonl；≈行 B 的 default 侧） | 1.00× |
| B. A + 问题类型重加权（`Mage-VL-QueryIP`，`keep_ratio=0.75`） | **62.3%** | 75.3% | 61.6% | 51.1% | **3703** | **0.75×**（相对同视频 default N） |
| C. B + QuoTA（可选） | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| D. B（或 C）前向 + Cultural LoRA | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

Δ(B−A)：overall **−1.4**；Short **−0.5**；Medium **−2.0**；Long **−1.6**。墙钟：行 B 推理约 **3.1 h**（含每题两次 codec：`measure` + `reweight`）。

### 5.1 行 B 读数与局限（2026-10-21）

- 714 题全出预测与打分；`qtype` 关键词分类：**dynamic 592 / static 122**（Video-MME 几乎总是「事件/动作」问法 → 重加权两档几乎退化成「常开 dynamic」）。
- Token 控制有效：`avg_n_vis_tokens / avg_default ≈ 0.750`（与 `keep_ratio` 一致）。
- **当前 A vs B 不是「同预算比 Acc」**：A 用满 codec 默认 patches，B 故意只留 75%。略降 Acc 换更少 token，符合效率叙事，**不能**写成「QueryIP 在同预算下涨点」。
- 若要严格消融「问题条件化是否有用」：应对齐 K（例如 A 也随机/按默认分数砍到 0.75K，再比 QueryIP 重加权）。

### 5.2 方案判断（结论写入计划）

1. **不公平对照（行 B vs 满预算 A）几乎没有 Acc 收益**：overall −1.4，同时省约 25% 视觉 token。  
2. **同 K 消融（§5.4）出现弱正信号**：短视频前 60 题上 queryip 略高于 random / prefix；**不足以**写成显著涨点。  
3. **动机（遗产静态题压运动）在本榜几乎测不到**：关键词把绝大多数题判成 dynamic；通用榜也不是本方法主战场（计划本就未承诺涨点）。  
4. **保留为「轻量接缝」合理**：不重训、预算可控、静/动选区可分；叙事应收束到「同预算下 query 条件化选留 + 可控减 token」，**不要**写成「Video-MME 已验证硬涨点 / 使用了 codec 真 MV」。  
5. PLAN 表里 Short **~75%** 是**满预算** Mage；同 K 砍到 0.75K 后 Acc 落到 ~57–62% **属预期**，勿与 75% 直接对比。

### 5.3 改正路线（2026-10-21 锁定）

| 路线 | 含义 | 状态 |
|------|------|------|
| **B（近）** | 承认 `cv-preinfer` 落盘**无** bitcost/MV 真分；canvas 内 patch 用帧差/纹理 proxy 打分并 top-K 打包；**同 K=0.75** 三路：`prefix` / `random` / `queryip` | **已完成** soccer `SAMEK_OK` + short-60（§5.4） |
| **A（远 · 延期）** | 改 Mage / `cv-preinfer`，把 readiness/bitcost **真分**导出，再按问题调权重选 patch；才对齐「用 Mage 的 MV/残差」 | **暂不实施**；启动前需确认：源码路径、能否改二进制并重装、是否允许改 `mage_vl`（原计划写不改） |

**路线 B soccer「成功」门槛（开放题，不算 Acc）：** 三路 `n_vis_tokens` 相同；QueryIP 的 kept 集合与 random、与 prefix **明显不同**；答案非空可读。→ **已达标**（`SAMEK_OK`，Jaccard≈0.60）。

**路线 B 诚实边界：** 仍是 proxy，论文**不能**硬写「使用了 codec 真 MV」。

建议消融命令见 [`COMMANDS.md`](../VQA/Explore-Then-Select/COMMANDS.md)（Agent 不代跑评测）。

### 5.4 路线 B 同 K 认证结果（2026-10-21）

**设定：** `keep_ratio=0.75`；三路共享同一 codec 编码与同一 `budget_k`；patch 级 proxy + 稠密打包（保证 tokens=features）。  
**产物：** [`Mage-VL-QueryIP_short60_samek.jsonl`](../VQA/eval_results/Mage-VL-QueryIP_short60_samek.jsonl)。

| 阶段 | 结果 |
|------|------|
| Soccer 同 K | `SAMEK_OK`：三路 `n_vis=3456` / `n_patches=13824`；QueryIP∩random / ∩prefix Jaccard≈0.60；答案非空 |
| Video-MME short **前 60 题**（20 视频） | 见下表 |

| 同 K 设置 | Acc | 题数 |
|-----------|-----|------|
| prefix（原序砍 K） | **56.7%** | 34/60 |
| random（seed=0） | **58.3%** | 35/60 |
| **queryip**（问题重加权） | **61.7%** | 37/60 |

Δ(queryip − random)=**+3.3 pp**（+2 题）；Δ(queryip − prefix)=**+5.0 pp**（+3 题）。

**读数注意：**

- 与 PLAN 行 A/B 的 Short **~75%** 不可比：那是满预算全 short 子集；本表是 **0.75K + 打包** + 仅前 60 题。  
- **n=60 抽样噪声大**；+2～3 题不足以宣称显著优势。  
- **判定：方法可保留**——作为轻量 query 条件化选留接缝，有同预算弱正信号；**未验证**通用榜硬涨点；若要更强证据，扩 short 全量或多 seed random，或走路线 A 真分。

---

## 6. 实验条件（单卡 4090D）

摘自 [`reproduction/README.md`](reproduction/README.md) 与 Cultural LoRA 计划。

| 项 | 值 |
|----|-----|
| GPU | NVIDIA GeForce RTX **4090 D 24GB** |
| 评测 / 训练环境 | `VLMEvalKit/vlmeval_env`（PyTorch 2.11+cu128 等，见 `envs/requirements-vlmevalkit.txt`） |
| 推理设定（对齐已有 Video-MME） | codec + HEVC；**32 frame**；`max_pixels ≈ 150000` |
| 行 B 预算 | `keep_ratio=0.75`：`budget_k = round(N_default × 0.75)` |
| LoRA 训练 knobs | bf16；gradient checkpointing；`max_seq_len=3072`；超长则减帧/降 `max_pixels`，**不裁答案** |
| LoRA 冒烟（参考） | 1000 step ≈ **11.5 min**；adapter ≈ **132 MB** |
| LoRA 全量 1 epoch（延期估） | **5–12 h** / 卡 |
| 新选区额外权重 | **无**（仅 α/β 规则重加权；不另存大模型） |
| 显存策略 | 24GB 内：推理按现网 Mage；训练开 grad ckpt；OOM 时降 `max_pixels` / 帧数 |

---

## 7. 文末注释（后续可再想）

### 7.1 明确不做：FlexSelect

- 固定总 token 下再按 LLM attention 精筛；在他模上有 Video-MME 涨点叙事，但接到 Mage 性价比低，且本计划不追通用榜硬涨点。
- GitHub（备查）：https://github.com/yunzhuzhang0918/flexselect

### 7.2 后续注释：ego-motion（本计划不做模块）

景区「运动≈人在动」是问题动机，可写进论文背景。  
**完整 ego-motion 估参 / 全局光流降权模块本计划不做**；是否另开设计留给后续再想。

### 7.3 与 Cultural LoRA 计划的互锁

[`PLAN_mage_cultural_lora_location.md`](PLAN_mage_cultural_lora_location.md) 已规定：不在默认 Mage 选区上跑正式全量 LoRA。  
本文件产出的新选区前向，才是正式 LoRA 与 Video-MME 复测的前提。

---

## 8. 阶段（摘要）

```text
[已完成] §0 soccer 冒烟 SMOKE_OK；行 B 全量 62.3% 已归档 → eval_results/Mage-VL-QueryIP_Video-MME.md
[进行中] 路线 B：patch 级 proxy + soccer 同 K 三路（prefix/random/queryip）
[下一步] 路线 B 成功后 → Video-MME short 前 60 题同 K Acc
[延期]   路线 A：导出 cv-preinfer 真分再重加权（需源码/可改二进制）
[可选]   QuoTA（表 §5 行 C）
[延期]   Cultural LoRA 全量（行 D）
[不做]   FlexSelect；新记忆/KV；完整 ego-motion 估参
```
