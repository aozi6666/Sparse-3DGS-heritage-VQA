# 工作 B 折中方案：Cultural LoRA + Oracle 位置 + Codec 效率表

> 状态：**方案已定，待实施**（非新架构；明确为导览场景的**域适配 + 系统融合**）。  
> 范围：Mage-VL 侧（`VQA/`）；本阶段**不改** Mage-ViT；**不改** 工作 A 的 3DGS 训练主线。  
> 总目标：公开榜有数字 + 系统有「位置条件问答」方法与验证表；时间紧时走薄 LoRA + Oracle + 加量 codec 表。  
> **后续（§12+）：** Question-aware codec 选区 + Ego-motion 补偿；**严格排在本阶段 P0 之后**；砍 DyCoke。  
> 上游：[`microsoft/Mage`](https://github.com/microsoft/Mage)；总叙事见 [`thesis.md`](thesis.md) / 仓外 `毕设.md`。  
> 与系统对接：[`system/README.md`](system/README.md)（定位与融合层此前未落地，本方案给出**可实现降级**）。

---

## 0. 贡献边界（写进论文时用）

| 算贡献 | 不算 / 勿吹 |
|--------|-------------|
| 文化遗产导览场景的 **LLM-side LoRA 域适配**（冻 ViT） | 从零重训 Mage-ViT / 换骨干 |
| **位置条件问答**（无位置 / Oracle / 估计）把 3DGS 资产焊到问答 | 复现 Mage 论文全部消融即「自己的方法」 |
| Codec 输入与分辨率等 **部署效率** 表（支撑实时导览） | patch 16/32/64（官方固定 16，不做主消融） |
| kitchen 上的**方法验证表**；遗产场景后补 demo | 把 Image VQA（CiQi-Bench）当视频导览主结论 |
| **（后续）** Question-aware codec 选区 +（可选）ego-motion 补偿 | DyCoke 等 LLM 侧再压 token；CoPE / Video-LaVIT 级重构 |

一句话：**先** Mage 看懂 + 薄 LoRA + 位置焊 3D + codec 效率表；**后** 只改进模前选区（问题感知 + 晃动补偿），不换编码器。

---

## 1. 已拍板决策

| 项 | 决定 |
|----|------|
| 场景 | **先 kitchen**（现成 3DGS）做方法验证表；**遗产场景后补** demo，正文注明 |
| Oracle prompt | **A** 区域名 + 短描述；**B** 区域名 + 该区说明卡全文（两档都做消融） |
| Codec 主刷 | **Video-MME 子集**上做完整消融矩阵；**OmniVDCBench `heritage_zh`** 只报默认配置 + 1–2 个关键对照 |
| 定位档位 | **P0 = 无位置 vs Oracle**；**P1 = 估计档（训练视角图像检索最近邻 → 继承区域标签）** |
| LoRA 训练 | **不用 ActivityNet**；**MSVD-QA 8k + CiQi-VQA 8k–16k**；1 epoch；**冻 ViT + LLM LoRA** |
| ChinaHeritaQA | **不进训练**；有余力时作图像辅测（**低优先级**） |
| 场景 QA | **接受 P0 约 30 题**（人工或模型出题 + 抽检）；题集服务方法表，**最终 demo 场景可换**，勿为 kitchen 过度投入 |

基线已有：Mage-VL × Video-MME（codec+HEVC，32frame）overall **63.7%**，见 `VQA/eval_results/Video-MME.md`。Humanity & History 样本稀，**只作辅证**。

---

## 2. 优先级总表

### P0（必须交差）

1. **公开域主测基线**：OmniVDCBench `heritage_zh` 上 Mage-VL Base（默认 codec 配置）。  
2. **薄 Cultural LoRA**：MSVD 8k + CiQi 子集 → 同协议复测 `heritage_zh` + Video-MME overall（及可选 H&H 辅行）。  
3. **位置条件问答（方法验证）**：kitchen 划 4–6 区 + 说明卡 + **~30 QA**；表：**无位置 / Oracle(短描述) / Oracle(说明卡)**。  
4. **Codec 效率主表（Video-MME）**：至少 Frames vs HEVC；补 `max_pixels` 与 `num_frames` 各一组合（见 §6）。记录 Accuracy / Tokens / Latency /（可选）显存。

### P1（强烈建议，时间允许）

1. **估计位置档**：query 帧 ↔ 训练/关键帧 **图像检索最近邻** → 继承该帧所属区域标签 → 注入与 Oracle 同格式 prompt；与 P0 三行对照成四行。  
2. **DCVC-RT** 一行接入 Frames/HEVC 对照（依赖环境；跑不通则附录说明）。  
3. **OmniVDCBench**：在 Base 与 LoRA 各用 **1 个默认 codec 配置** 报 Acc；另选 1 个对照（如 Frames vs HEVC，或 150k vs 300k `max_pixels`）。  
4. CiQi-Bench 图像辅测（证明领域知识，**非主结论**）。

### P2（低优先级 / 填充工作量）

- Video-MME 上加长：时长桶 × backend、更密的 `num_frames` / `max_pixels` 曲线。  
- ChinaHeritaQA 图像辅测。  
- 遗产自采场景：复用同一套区域卡 + QA 协议做 demo（**不要求**与 kitchen 同表规模）。  
- 完整 GSVisLoc 替换检索最近邻（**无官方开源则不做**；论文写「降级实现，接口预留」）。

---

## 3. 数据方案

### 3.1 LoRA 训练（主配方，已定）

| 来源 | 角色 | 用量 | 备注 |
|------|------|------|------|
| [MSVD-QA](https://github.com/xudejing/video-question-answering) | 短视频问答、稳住视频能力 | **8k** 条 | 英文为主；视频易获取 |
| [CiQi-VQA SFT](https://huggingface.co/datasets/SII-Monument-Valley/CiQi-VQA) | 文物/瓷器领域知识 | **8k–16k**（随机或按属性分层） | 全库约 55.7 万，**禁止全量** |
| ActivityNet-QA | — | **不用** | 下载死链多、耗时 |

合计约 **16k–24k** 样本，**1 epoch**。配比叙事：MSVD 保视频问答格式；CiQi 拉文化遗产术语与属性（朝代/釉色/纹饰等）。域差如实写：瓷器图 ≠ 非遗视频，属**弱监督领域迁移**。

**不做：** ChinaHeritaQA 进训练（其定位更偏 benchmark，且 HF gated）。

### 3.2 评测集

| 集合 | 用途 | 优先级 |
|------|------|--------|
| OmniVDCBench `heritage_zh`（330 视频 / 1244 QA） | **领域视频主测** | P0 |
| Video-MME（已有链路） | 通用视频 + **codec 消融主阵地** | P0 |
| CiQi-Bench | 图像领域辅测 | P1 |
| ChinaHeritaQA | 图像辅测 | P2 |
| kitchen 自建 ~30 QA | **位置条件**方法表 | P0 |
| Video-MME Humanity & History | 辅行（样本少） | 辅证 |

训练与评测隔离：CiQi 子集抽样时 **排除 CiQi-Bench 标本**；OmniVDCBench / Video-MME **零样本评测**。

### 3.3 场景内 QA（kitchen 方法验证）

- 区域数：**4–6**；每区：`region_id`、短描述（≤1–2 句）、说明卡（一段知识文本）。  
- 题量：**~30**（每区约 5–8 题）；题型建议：区域内可见物、属性、需说明卡才答得出的知识题（用于拉开 Oracle 档）。  
- 出题：**模型根据说明卡草稿 + 人工抽检**（省时间）；标注每题绑定 `region_id`。  
- 视频/帧：用该区域对应训练视角或渲染视角短片段即可，**不追求导览级采集**。  
- 正文写明：定量协议在 kitchen 验证方法；**最终系统 demo 可换遗产场景**，复用同一 prompt 协议。

---

## 4. 训练设置与单卡 4090 时间（估算）

### 4.1 方法设定

- 基座：`microsoft/Mage-VL`（Qwen3-4B 解码 + Mage-ViT）。  
- **冻结** Mage-ViT（及视觉侧）；**仅 LLM（及必要 connector，若官方默认冻结则保持冻结）上挂 LoRA**。  
- LoRA 建议起点：`r=16`，`alpha=32`，dropout `0.05`；目标模块：注意力 / MLP 线性层（实现时按 PEFT 可挂模块微调）。  
- 优化：1 epoch；lr ~1e-4–2e-4；cosine；warmup ~3%；有效 batch ~8（`per_device=1` + grad_acc）。  
- 视频：与推理接近的帧数（如 8–16）与 `max_pixels≈150k`，避免训推严重错配。  
- **叙事**：域适配，不是新架构。

> Mage 官方以推理发布为主；需自建 SFT/PEFT 数据管线（对话格式对齐 `mage_vl` processor）。此为 P0 工程项。

### 4.2 时间量级（1× RTX 4090 24GB，经验区间）

| 配方 | 样本 | 约 1 epoch 墙钟 |
|------|------|-----------------|
| CiQi 10k（仅图） | 10k | ~6–12 h |
| **MSVD 8k + CiQi 10k（默认）** | ~18k | **~1–2 天** |
| MSVD 8k + CiQi 16k | ~24k | ~1.5–2.5 天 |
| CiQi 全量 55 万 | — | **不做** |

另计：下载与转格式、评测各半日到数日。优先保证 **可复现的小配比跑通**，再视曲线加 CiQi 到 16k。

---

## 5. 定位降级设计（GSVisLoc 无开源）

目标接口不变：`query_frame (+ 可选 3DGS) → region_id / 区域文本 → 拼进 Mage prompt`。

```text
Query RGB
   │
   ├─ P0 Oracle：人工/标注给定 region_id → 查表短描述 / 说明卡
   │
   └─ P1 Estimate：图像编码器检索训练集/关键帧库
         → Top-1 帧的预标 region_id
         → 同 Oracle 的文本注入
```

| 档位 | 实现 | 何时做 |
|------|------|--------|
| None | 仅视频/图像 + 问题 | P0 |
| Oracle-short | 区域名 + 短描述 | P0 |
| Oracle-card | 区域名 + 说明卡全文 | P0 |
| Est-NN | 全局描述子（可选 CLIP/DINOv2）对关键帧库检索，继承区域标签 | P1 |
| GSVisLoc 级 | 3D–2D matching + PnP + 渲染 refine | **P2 / 不做**（无代码则论文称未来工作） |

**稳定优先：** P1 不依赖可微渲染闭环；区域标签做在**关键帧**上即可（每区若干张），避免稠密三维标注。  
估计档指标除 Acc 外可报：**区域 Top-1 命中率**（检索是否找对区），与问答 Acc 分开写，避免混淆。

Prompt 拼接示例（实现时统一模板）：

```text
[Location] 区域：{region_name}。{short_or_card}
[Question] {user_question}
```

---

## 6. Codec / 推理消融（撑表，零训练）

官方已支持：`frames` / `codec+HEVC` / `codec+DCVC-RT`、`num_frames`、`max_pixels`；**patch=16 固定，不扫 patch。**

### 6.1 在 Video-MME 上的矩阵（主表，可加量）

| 实验 | 变量 | 记录 |
|------|------|------|
| A 输入方式 | Frames / HEVC /（P1）DCVC-RT | Acc, tokens, latency, mem |
| B 帧数 | 16 / 32 / 64 | Acc, tokens, latency |
| C 分辨率 | `max_pixels` 75k / 150k / 300k | Acc, tokens, latency, mem |
| D（P2）时长 | 按 Video-MME short/medium/long 切，或自截 10/30/60s | Acc vs backend |

默认锚点：与现有结果可比的 **HEVC + 32 frame + 既有 max_pixels**。LoRA 后模型固定再扫一遍关键子集（不必全矩阵×两模型；P0 可 Base 全扫 + LoRA 只扫 A/C 默认点）。

### 6.2 在 OmniVDCBench `heritage_zh` 上（辅表）

- Base + LoRA，**同一默认配置**各一行 Acc。  
- 再加 **1–2** 对照（推荐：Frames vs HEVC，或 150k vs 300k）。  
- 不在此刷满 B/C/D 全网格（太慢）。

---

## 7. 结果表骨架（论文用）

**表 1 — 域适配（P0）**

| 方法 | OmniVDCBench Heritage | Video-MME Overall |
|------|----------------------|-------------------|
| Mage-VL Base | Acc | Acc |
| Mage-VL + Cultural LoRA | Acc | Acc |

**表 2 — 位置条件（P0，kitchen 协议；P1 加 Est）**

| 方法 | Acc | 区域命中率（仅 Est） |
|------|-----|----------------------|
| Base / LoRA + 无位置 | | — |
| + Oracle-short | | — |
| + Oracle-card | | — |
| + Est-NN（P1） | | Top-1 |

**表 3+ — Codec（Video-MME；行数可加）**  
按 §6.1 展开；强调 tokens/latency 与 Acc 权衡，服务「实时导览」叙事。

**辅表：** CiQi-Bench；ChinaHeritaQA（P2）；Video-MME H&H。

---

## 8. 实施顺序（省时间）

```text
1) 下 OmniVDCBench heritage_zh → Base 评测          [P0]
2) 备 MSVD 8k + CiQi 8k–10k → LoRA 1 epoch         [P0]
3) LoRA 复测 heritage_zh + Video-MME               [P0]
4) kitchen 4–6 区说明卡 + ~30 QA → Oracle 表       [P0]
5) Video-MME：Frames/HEVC + frames + max_pixels    [P0]
6) Est-NN 检索档 +（可选）DCVC + OmniVDC 1–2 对照 [P1]
7) 遗产场景 demo / ChinaHeritaQA / 加长曲线         [P2]
```

---

## 9. 风险与不做清单

| 风险 | 应对 |
|------|------|
| Mage 无官方 SFT 脚本 | 自接 PEFT；先小子集（1k）过拟合冒烟再全配比 |
| CiQi→非遗视频迁移弱 | 主结论看 OmniVDCBench 相对 Base 的 Δ；弱则强调位置模块与系统 |
| 场景 QA 过少方差大 | 30 题报 overall + 分「需知识卡 / 不需」子集；不夸大显著性 |
| DCVC 环境失败 | 主表保留 Frames/HEVC；DCVC 附录 |
| demo 换场景 | 协议与代码按 region 卡配置化，不写死 kitchen 文案 |

**明确不做（本阶段 + 后续选区阶段共用）：** patch 16/32/64 消融；AN-QA 训练；ChinaHeritaQA 主训练；全量 CiQi；无开源前提下的完整 GSVisLoc 复现；以 Image VQA 替代视频主测；**DyCoke / FastV / FrameFusion 等 LLM 侧再压 token**（与 Mage 进模前选区重复，时间紧砍掉）；CoPE-VideoLM / Video-LaVIT 级换栈。

---

## 10. 成功标准（可答辩）— 本阶段（Cultural + Location）

- OmniVDCBench Heritage：LoRA **不低于 Base**，争取稳定小幅提升（如实报）。  
- kitchen 位置表：Oracle-card **明显高于**无位置（尤其知识题子集）。  
- Codec 表：HEVC（或 DCVC）相对 Frames 在 tokens/latency 上有清晰优势，Acc 可接受。  
- 论文表述无「提出 Mage 新骨干」；本阶段贡献落在 **域适配 + 位置条件融合 + 效率验证**。

---

## 11. 相关路径

| 路径 | 说明 |
|------|------|
| `VQA/mage_vl/` | 推理 / streaming；`codec_config`、`max_pixels`、`num_frames` |
| `VQA/VLMEvalKit/` | Video-MME 等评测 |
| `VQA/eval_results/` | 已有 Mage × Video-MME 指标 |
| `3DGS/.../data/mip360/kitchen` | Oracle / Est 方法验证用资产 |
| [`system/README.md`](system/README.md) | 系统三路对接；本方案补定位降级 |

**执行纪律：** 必须先完成上文 **P0（Cultural LoRA + Oracle + Video-MME codec 主表）**，再启动下面 **§12 后续工作**。

---

# 后续工作：Question-aware Codec 选区 + Ego-motion 补偿

> 状态：**已定方案，排在 Cultural LoRA + Oracle 之后**（严格先后，不抢本阶段工期）。  
> 目标：在 **不换 Mage-ViT / 不重构 codec 接口** 的前提下，改进 **P 帧 patch 打分与 top-k**，争取 OmniVDCBench / Video-MME 涨点或同点加速。  
> 约束：无自建训练数据；1×4090；优先零训练，掉点再薄 LoRA。

---

## 12. 后续工作优先级

| 级 | 内容 | 说明 |
|----|------|------|
| **前提** | 本文件 §2 P0 全部完成 | Cultural LoRA + Oracle 位置表 + Video-MME codec 主表 |
| **F-P0** | **Question-aware codec selection（零训练）** | 主方法；评测用公开题自带 question |
| **F-P0.5** | 同协议复测；若 Acc **明显掉点** → **+ 薄 LoRA** 第五行 | 数据仍用 MSVD 8k + CiQi 8–16k；先写进消融表，不够再跑 |
| **F-P1** | **Ego-motion 补偿后再算 MV**（OneVision 思路） | 小改进 / 消融一行；服务手持晃动，**不是**主答指令手段 |
| **F-砍** | DyCoke / 同类 LLM 侧动态压 token | 与 Mage MV+residual「进模前少选」重复，不做 |

相对叙事：

- **Question-aware** → 对准「答问题」：少选与问题无关的运动区，多留相关 patch（含偏静态但语义相关区域，靠 CLIP 相关分补）。  
- **Ego-motion** → 对准「导览晃动」：全局相机运动从 MV 里剥掉，少把背景当重要运动；对纹饰/碑文等**纯静态**问答帮助有限。

---

## 13. 参考项目（带源码，实现时对齐思路）

| 用途 | 项目 | 论文 | 代码 |
|------|------|------|------|
| 基座（已用） | Mage / Mage-VL | [arXiv:2607.24904](https://arxiv.org/abs/2607.24904) | https://github.com/microsoft/Mage |
| 问题引导稀疏（灵感：文本相关再剪/重排） | SparseVLM | [arXiv:2410.04417](https://arxiv.org/abs/2410.04417) | https://github.com/Gumpest/SparseVLMs |
| Ego-motion + MV/residual 选区 | OneVision-Encoder | [arXiv html](https://arxiv.org/html/2602.08683v3) | https://github.com/EvolvingLMMs-Lab/OneVision-Encoder |
| 问题相关分编码器 | OpenAI CLIP | [Radford et al.](https://arxiv.org/abs/2103.00020) | HF：`openai/clip-vit-large-patch14`（或 `openai/clip-vit-base-patch32` 省显存） |

**不做但需在相关工作提及：** [DyCoke](https://github.com/KD-TAO/DyCoke)（CVPR’25，LLM/KV 侧压 token）— 本方案明确排除。

Mage 自身已基于 codec-aligned sparsity（与 OneVision 同族）；本后续工作是 **选区策略插件**，不是替换 Mage-ViT。

---

## 14. 实现主要思路

### 14.1 共同点（相对原版 Mage）

原版：HEVC（或 DCVC）→ 每 patch `score_codec = f(MV, residual)` → P 帧 top-k → Mage-ViT。  
后续：**只改 `score` 与（可选）MV 预处理**；patch 网格仍 16×16；ViT/LLM 权重默认不动。

```text
Video + Question
    │
    ├─ HEVC/DCVC → MV, residual
    │       │
    │       └─ [F-P1] Ego-motion 估计（中位/相似/仿射）→ MV_obj = MV - MV_cam
    │
    ├─ score_codec(patch) ← |MV_obj| + residual energy   （与 Mage/OneVision 一致族）
    ├─ score_q(patch)     ← CLIP(text=question, image=patch或邻域)
    │
    └─ score = normalize(score_codec) + λ · normalize(score_q)
              → P-frame top-k → 原 Mage-ViT →（可选薄 LoRA）LLM
```

`λ`、融合用加性还是乘性：实现时用验证子集扫 2–3 个值即可，论文报默认 `λ`。

### 14.2 Question-aware（F-P0，必做）

**实现方式**

1. 在 `VQA/mage_vl` 选 patch 路径（`codec_config` / processor 上游）接入钩子：输入增加 `question: str`。  
2. 对每个候选 P-patch（或先按 codec 粗排取 top-M，再 CLIP 精排 top-k，省算力）：裁 RGB → CLIP image tower；question → CLIP text tower → 余弦相似度作 `score_q`。  
3. 与归一化后的 `score_codec` 融合，按融合分取 top-k（k 与原 Mage token budget 对齐，保证可比）。  
4. I 帧仍全保留（与 Mage 一致），只动 P 帧。  
5. 零训练推理；Flash / 批量化 CLIP 编码，避免评测墙钟爆炸。

**提供什么**

- 配置项：如 `--patch-score question_aware`、`--clip-model openai/clip-vit-large-patch14`、`--lambda-q`。  
- 相对 SparseVLM：对方多在 **LLM 注意力层后** 剪 token；本方案在 **codec 选区前** 融入问题，更贴「改进 Mage codec-native」叙事，且复用 SparseVLM「文本应引导视觉稀疏」的思想（cite SparseVLM + Mage）。

**薄 LoRA（F-P0.5，表格先留行）**

- 条件：零训练 Question-aware 在 OmniVDCBench 上相对 Base **明显掉点**（建议阈值：overall 掉 ≥1–2 个点再开；具体阈值实现时定）。  
- 做法：冻 ViT；LLM LoRA；数据仍 **MSVD 8k + CiQi 8–16k**；1 epoch；训练时 **前向必须走同一套 question-aware 选区**（训推一致）。  
- 消融表预留第 5 行「+ QA-select + 薄 LoRA」；若不掉点则该行标注「未触发 / N/A」。

### 14.3 Ego-motion 补偿（F-P1）

**解决什么 / 不解决什么**

- **解决：** 手持、跟拍、运镜时，全局相机 MV 污染 → 背景被当成“重要运动” → token 浪费在背景。补偿后更偏向物体运动。  
- **不解决：** 纹饰/碑文等几乎无 MV 的静态细节问答（那更靠 residual + Question-aware 的 CLIP 分）。  
- **定位：** 小改进、消融一行，**不当主贡献**。

**实现方式**

- 对照 OneVision-Encoder 仓库中 camera motion compensation（median / similarity / affine）对 MV 的处理，在 Mage 提 MV 之后、聚合成 patch score 之前插入。  
- 参考入口：https://github.com/EvolvingLMMs-Lab/OneVision-Encoder （README / codec-style patch、MV 分析说明）；论文 [html](https://arxiv.org/html/2602.08683v3)。  
- 默认先试 **median**（最稳、最好实现），sim/affine 作可选消融。

**提供什么**

- 开关：`--mv-ego-comp {off,median,similarity,affine}`。  
- 可与 Question-aware 正交组合（先补偿 MV，再算 codec 分 + CLIP 分）。

---

## 15. 后续测评指标

| 指标 | 含义 | 主看 |
|------|------|------|
| **Accuracy** | OmniVDCBench `heritage_zh`；Video-MME overall（及 short/medium/long） | 涨点主指标 |
| **Visual tokens / video** | 平均视觉 token 数（与 Base 对齐 budget 时主要看 Acc；若扫 k 则看 Acc–token 曲线） | 效率 |
| **Latency** | 单题或单视频墙钟（含 CLIP 开销要单列或注明） | 部署 |
| **GPU mem** | 峰值显存（可选） | 附录 |
| **选区诊断（建议）** | 定性：同帧上 codec-only vs +Q vs +ego 的 patch 热力/叠加图 | 论文图 |

主测协议与本阶段一致：OmniVDCBench 报默认 codec 配置；Video-MME 可做选区方法对照（不必重刷全部 frames/max_pixels 网格）。

---

## 16. 消融表骨架（后续章）

固定 Mage-VL 权重（及本阶段已训好的 Cultural LoRA **二选一写清**：默认先在 **Base** 上扫选区；确认增益后再挂到 Cultural LoRA 上各报一行或一组）。

| ID | 方法 | OmniVDC Heritage Acc | Video-MME Acc | Tokens | Latency |
|----|------|----------------------|---------------|--------|---------|
| M0 | Mage 原版 MV+residual | | | | |
| M1 | + Question-aware（CLIP，零训练） | | | | |
| M2 | + Ego-motion（median） | | | | |
| M3 | + Question-aware + Ego-motion | | | | |
| M4 | + Question-aware + 薄 LoRA | | | | |  ← **预留**；零训练不掉点则标 N/A，不跑 |

可选附录：`λ` 扫描；ego 的 median vs affine；CLIP ViT-L vs ViT-B。

---

## 17. 实验图（后续章建议产出）

| 图 | 内容 |
|----|------|
| 图 F1 | 流水线：MV/residual →（ego）→ codec 分 ⊕ CLIP(question) → top-k → Mage-ViT |
| 图 F2 | 定性选区对比：同一帧 / 同一问题下 M0 vs M1 vs M3 的保留 patch 可视化（导览晃动例 + 静态细节例各 1） |
| 图 F3 | OmniVDCBench / Video-MME：M0–M3（及 M4 若触发）柱状 Acc |
| 图 F4 |（可选）Acc vs visual tokens 或 vs latency（若扫 k 或 λ） |

表：§16 主消融表 +（可选）λ / ego 变体附录表。

---

## 18. 后续实施顺序与准备清单

```text
[本阶段 P0 全部完成]
    → F1 接 CLIP + question-aware 钩子，小子集冒烟
    → F2 OmniVDCBench + Video-MME：M0 vs M1          [F-P0]
    → F3 若掉点 → 薄 LoRA → M4                        [F-P0.5]
    → F4 接 ego-motion median → M2 / M3               [F-P1]
    → F5 选区可视化 + 写章节
```

**需要准备（无自建训练集）**

| 项 | 说明 |
|----|------|
| Mage-VL 权重 + 本阶段评测脚本 | 已有 |
| OmniVDCBench `heritage_zh`、Video-MME | 评测；question 来自标注 |
| `openai/clip-vit-large-patch14`（或 B/32） | 问题相关分；HF 下载 |
| OneVision-Encoder 仓库（只读参考 MV 补偿） | clone 对照，不必替换 Mage 骨干 |
| （条件触发）MSVD 8k + CiQi 子集 | 仅 M4 薄 LoRA |

**成功标准（后续）**

- M1 或 M3 相对 M0：OmniVDCBench Acc **不降优先，争取提升**；Latency 增幅可接受（CLIP 开销写明）。  
- M2 在「运镜/晃动」样例上选区更合理（图 F2）；静态题不指望 M2 独涨。  
- 论文表述：改进 **codec-native patch selection**（问题感知 + 可选相机运动补偿），cite SparseVLM / OneVision / Mage；**不**宣称新视觉骨干。
