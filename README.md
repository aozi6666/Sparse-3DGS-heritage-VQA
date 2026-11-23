> 分支同步（2026-11-20）：[旧分支目录回退原因与同步规则](docs/BRANCH_SYNC_20261120.md)。主线目录统一为VLM，独立历史实验分支保留各自源码。

> 当前入口（2026-11-20）：[main统一整理、VGGT保全与分割支线退出](docs/MAIN_MIGRATION_20261120.md)。视频工程仅使用 `VLM/`；VCAR已取消。数据、权重、环境留在服务器，不随Git分发。

# Sparse-3DGS-heritage-VQA

## 视频理解：无字幕画布替换与字幕增强

本项目分别评估两种离线问答方法。视觉路线研究固定视觉网格下如何分配完整帧与运动区域；字幕路线研究如何利用与画面时间对齐的语言证据。以下数字来自本地 VideoMME 子集（238 视频、714 题）。

### 问题相关完整帧与 Mage-VL 运动画布

Mage-VL 根据原视频编码信息选择 patch，按目标 32 个 codec 画布组织输入。AKS 独立读取原视频候选帧与问题的 BLIP ITM Large 预计算分数和 32 帧入选结果。将候选按四个时间区间排序，保护首个有效画布，以完整 RGB 帧替换源时间相近的整个画布，同时更新时空坐标。不同上限使用嵌套位置，实际替换量取决于有效画布数及时间碰撞。

完整预算扫描已完成。无外部字幕的固定 K=20 答对 **485/714（67.93%）**，较原版 455/714 净增 30 题。研究目标为减少语义帧替换，同时保持问答准确率。短 4、中长 20 的事后回放同为 485/714，平均替换数从 13.74 降至 10.50。独立视频验证及端到端计时尚待完成。

[方法、对照实验与结论](docs/experiments/aks_selection/README.md) · [十组预算数据](docs/experiments/aks_selection/all_budgets.csv) · [逐题精简结果](docs/experiments/aks_selection/paired.csv) · [最终实验代码](VLM/aks_mage_budget_final/sweep.py)。

字幕路线请参见下节。字幕实验有自己的无字幕对照（459/714），视觉预算实验使用 455/714；各自沿用对应协议计算增益。两条路线的联合效果尚待验证。

## 当前成果：原生Codec＋完整时间对齐字幕，714题已完成（2026-11-06）

**保留“原生Codec＋完整时间对齐字幕”成果，进入归档、制表和画图阶段，不重跑714。** 本次收益属于字幕证据增强；A.I.R.问题相关时间选择仍未证明稳定额外收益，保留代码和负结果并暂停。当前没有接入A.I.R.、API判决器或新增训练。

**实际接入流程：** 视频经Mage原生Codec生成canvas及来源patch位置／时间；根据实际保留patch的来源时刻匹配真实字幕，补充既定相邻句、去重并添加时间标记；视觉与字幕共同输入原Mage问答。原生Codec的视觉patch、模型权重及QA配置保持不变，Codec本身归属原Mage方法，不作为本项目原创。

### 本地714题结果

714题／238视频的三路Codec配对共2142/2142单元完成，714题输入审计通过、非法答案0；180题历史无字幕控制核对通过。开发阶段540个旧答案在实际Tensor一致后复用，其余同题同输入也按冻结规则复用。已有Frames结果保留作参考。

- 原生Codec无字幕：**459/714，64.29%**。
- Codec＋最多32个字幕锚点／2048对照：**504/714，70.59%**。
- **Codec＋完整时间对齐字幕：521/714，72.97%**。
- 已有Frames-32＋连续时间对齐字幕2048：**511/714，71.57%**。

完整字幕相对原生Codec无字幕，**救回92题、误伤30题，净增62题／8.68个百分点**；视频聚类bootstrap 95%区间为**[5.74，11.62]个百分点**，支持本批数据上的正向收益。整体增益包含额外字幕证据和文本token成本，不能归因于新增选区算法。

分时长的Codec无字幕→完整字幕结果：

- 短视频：**166/219，75.80% → 176/219，80.37%**，净增10题／4.57个百分点。
- 中等视频：**162/258，62.79% → 183/258，70.93%**，净增21题／8.14个百分点。
- 长视频：**131/237，55.27% → 162/237，68.35%**，净增31题／13.08个百分点；仍存在逐题误伤，不能写成“没有误伤”。

### 比较边界与结论

- 相对32字幕锚点／2048对照，完整字幕净增17题／2.38个百分点，整体聚类95%区间[0.28，4.62]。此比较同时改变锚点覆盖与文本预算，**不是单独的预算消融**；完整字幕、补句和时间标记的贡献不能混为一个已证实的新算法。
- 相对Frames-32＋连续字幕，整体只净增10题／1.40个百分点，95%区间**[-1.26，4.20]**跨零，**尚不能宣称稳定优于Frames方案**。长视频多对18题，中等视频少对9题，短视频多对1题；不依据这些事后结果按时长拼接最优路线。
- 历史test225中，Codec无字幕**131/225，58.22%**，完整字幕**158/225，70.22%**，救回35题、误伤8题、净增27题／12个百分点；Frames连续字幕为**163/225，72.44%**，比本次Codec完整字幕多对5题。保留两条路线，如实报告差异。
- 714由short219、dev180、val90、test225四阶段组成，包含开发和已参与历史诊断的样本；不是官方Video-MME 2700题完整基准，也不是714题全新独立测试。开发、补充验证、历史测试应分开报告，聚类区间不消除历史选型／诊断影响。不能直接与原论文官方全量分数作同协议排名。

### 冻结配置、数据与后续工作

主路线统一字幕上限**32768 token**，全部238视频实际最大字幕为**14217 token**；不填充文本。32字幕锚点／2048对照及Codec视觉配置固定。旧13312阶段因单视频超预算停止的现场、237个已通过记录及所有缓存保留原位；缓存未重编码，237个旧视频实际完整字幕和控制证据已核验不变。放宽上限本身没有让其余输入变长。

本次新生成1296次、复用846单元，新生成累计耗时1331.47秒、实测峰值显存13.44GiB；这些是带缓存复用的实验成本，**不是部署端到端耗时**，不能直接与原论文Table 5的8×B200墙钟时间比较。

- [完整汇总](VLM/eval_results/practical_mage_v1/24_codec_full714_32768_v1/results/full714/summary.json)、[逐题预测](VLM/eval_results/practical_mage_v1/24_codec_full714_32768_v1/results/full714/predictions.csv)、[分阶段／分时长准确率](VLM/eval_results/practical_mage_v1/24_codec_full714_32768_v1/results/full714/metrics.csv)、[救回误伤及区间](VLM/eval_results/practical_mage_v1/24_codec_full714_32768_v1/results/full714/paired_comparisons.csv)、[token与推理成本](VLM/eval_results/practical_mage_v1/24_codec_full714_32768_v1/results/full714/token_cost.csv)。
- 全部新成果位于`VLM/eval_results/practical_mage_v1/24_codec_full714_32768_v1/`；协议、缓存来源、输入审计、断点、日志、原始回答及旧结果均保留，后续画图直接读取上述CSV／JSON。
- 执行入口`VLM/mage_vl/codec_full714_32768/run_pipeline.sh`保留用于追溯和中断恢复，**已完成的714题无需再次运行**。[32768计划及协议说明](docs/PLAN_codec_full714_fixed32768.md)、[旧13312冻结计划](docs/PLAN_codec_full714_fixed13312.md)。
- 下一步只做成果归档、主结果表、分时长准确率图、救回／误伤图、token成本图及成功／失败案例。表格分别标明本地714、各阶段角色、模型、视觉预算、字幕覆盖／文本预算与额外成本，保留已有字幕消融和A.I.R.负结果。

以下“待执行”等段落均为历史计划；当前结论以本节及实际汇总为准。

## 历史计划：Codec开发配对（2026-11-06）

用户选择完整字幕证据优先；60视频最大完整字幕13043 token，全体主路线统一13312，对照仍32字幕锚点/2048，不改视觉patch。先对最长输入702—704进行显存门禁探测，通过后跑dev180，探测答案计入开发阶段。旧6144冒烟、字幕成绩及负结果保留。结果存22_codec_dev180_v1，见[计划](docs/PLAN_codec_dev180_complete13312.md)。新GPU成绩尚未产生。


## 历史计划：Codec字幕冒烟（2026-11-06）

固定短中长9题：原生Codec无字幕、全部保留来源时刻＋6144字幕、最多32个字幕锚点＋2048对照，保留旧Frames字幕参考。扩大字幕预算只因长视频完整证据需要5147 token，不代表已证明更好；三路视觉patch不变。14项CPU测试与9题真实旧canvas输入审计通过，尚未启动真实QA。只读复用旧缓存，断点/日志/数据存21_codec_subtitles_v1。见[计划与运行命令](docs/PLAN_codec_subtitles_smoke9.md)。


## 历史计划：字幕消融准备状态（2026-11-06）

A.I.R.语义时间选择与视觉细节增强暂停，代码/负结果保留；不将其作为已验证有效模块。保留Frames-32＋连续时间对齐字幕2048成果。本轮只补齐714题的去时间标记、去相邻补句两项固定消融，三路旧结果不重跑；Codec＋字幕另开阶段，尚未验证。已完成18项CPU测试、3项原生处理器合成RGB检查及714题来源预检，真实GPU实验交用户执行，当前没有新准确率成绩。

最多新增1101次生成，327个相同提示在输入核对后复用；结果和绘图数据保存到 VLM/eval_results/practical_mage_v1/20_subtitle_ablation_v1/frozen_r1/。全量指本地714题，不是官方2700题；新消融为事后分析，不重新选型。详见[实施计划与运行说明](docs/PLAN_subtitle_ablation_full714.md)。



### 字幕收尾消融已完成（2026-11-06）

1428单元审计通过、非法0。714题五路无字幕/原对齐/连续/去标记/去邻句分别474/496/511/509/502；原test225分别140/154/163/162/158。整体字幕收益保留，补句正向趋势、时间标记边际弱，组件区间均跨零，不宣称独立显著。停止字幕调参，Codec另作接口/预算冒烟，不直接跑全量。详见[实际结论](docs/ANALYSIS_subtitle_ablation_full714_20261106.md)。

## 历史研究设计：问答总设计与优先级（2026-11-05）

总设计为**问题条件化时间选择＋Mage原生Codec稀疏视觉表示＋时间对齐字幕证据**。以可重复的实用涨点为目标，不要求强算法创新；当前是拟议联合设计，不是已验证的三模块融合系统。

- **语义时间选择**决定重点处理哪些时间范围，必须考虑目标、限定条件、事件覆盖及前后关系；相关画面不等于满足完整题意的证据。
- **Mage原生Codec**组织稀疏视觉信息，涉及canvas、来源patch位置及时间信息，再进入Mage视觉编码和多模态解码。不是单纯图内空间选区；本轮Frames语义选帧没有修改原生Codec或模型权重。
- **时间对齐字幕**补充对应解说，联合时必须按实际视觉输入时间重新匹配，固定2048字幕token不能冒充总输入预算固定。

不能假定硬裁片段后再Codec天然优于原生全视频Codec：硬裁可能丢全局证据、改变编码结构；target_canvas不是帧数。片段边界、原时间映射、patch位置及提示时间、字幕时间轴和实际token预算都需要验证。保留Codec，但不为架构完整强行将语义选择放在它前面。

**阶段准入顺序：语义选帧先证明新增题收益 → 验证Codec接口和无字幕增量 → 最后验证字幕组合。** 当前去观察查询候选在开发30题为18/30（均匀16/30），新增18题与均匀均12/18；尚无稳定收益证据。旧规则和失败结果已冻结归档，新的研究必须独立版本，不覆盖旧成果，不在已见题上反复挑阈值制造涨点。

优先攻克的三个瓶颈：

1. **完整条件定位**：绳子出现不代表用于第一个魔术；中途领先不代表最终获胜；顺序/计数需要找齐相关事件。CLIP相似度仅是相关性线索，不是充分性证明。
2. **局部强化与全局覆盖**：增加局部片段会挤掉其他证据；首次、最终、否定、顺序、计数题可能受损。全局锚点只是保护措施，不保证完整性。
3. **片段内时间分辨率**：12秒内仅4帧可能漏掉瞬时动作。必须区分“定位错”“片段对但采样漏动作”“证据到位但Mage判断错”，不能用统一阈值处理所有失败。

最终联合评测至少保留四组：原生Codec；原生Codec＋字幕；语义时间选择＋Codec；语义时间选择＋Codec＋字幕。另保留已有**Frames-32＋连续字幕2048**作为实际参考，并以同片段数量/时长/视觉预算的均匀时间选择控制语义贡献。若联合系统不能超过实际参考，或相近效果代价明显增加，就不采用复杂版本。必要时固定文本证据做诊断，以区分视觉改进和字幕内容变化。

字幕已有冻结test225结果：无字幕140/225、连续字幕163/225；不能把该提升与选帧局部提升相加。当前两模块分别实现、分别评测，尚未联合接入Codec。旧Query-IP入口已在本分支清理，下文相关指标仅作历史记录，不是当前语义选帧成果。重建与文旅交互闭环仍需各自证据，不由公开视频QA结果替代。

本轮源码检索与CPU诊断见[语义证据研究记录](docs/RESEARCH_semantic_evidence_next_20261105.md)：官方时间标注覆盖13/30题，仅作离线覆盖诊断；AKS默认影子选帧没有新增QA成绩。

新探索工程：[HiMu启发的条件解析＋子事件定位](docs/PLAN_condition_grounding_v1.md)已完成CPU准备，先做5题纯文本解析实测；尚无新定位或QA效果，不改变旧候选归档结论。

当前归档：[语义选帧阶段总结](docs/SUMMARY_air_semantic_frames_archive_20261105.md)、[字幕冻结结果](docs/SUMMARY_mage_subtitle_final_results_20261103.md)。所有新实验记录、日志、断点和新缓存留在`VLM/eval_results/practical_mage_v1/`；旧数据和缓存原位保留。助手默认只进行源码/只读分析与CPU验证，GPU推理命令交用户执行；不读取API密钥，不自动训练/下载。


> 论文题目：**受限采集下的 3DGS 三维重建方法以及在文化遗产数字展示中的应用研究**  
> 全文设计与流程图见 [`docs/thesis.md`](docs/thesis.md)。

## 论文目标（摘要）

工作 A 从稀疏视角构建可自由观看的 3DGS 数字资产；工作 B 对离线视频进行问答，分别评估视觉取帧与时间对齐字幕的贡献。

| 工作 | 内容 | 本仓状态 |
|------|------|----------|
| **A · 稀疏 3DGS 重建** | init（classic / VGGT±depthft）→ Drop 粗训 → LoRA repair（± CADR / 底模） | **消融已齐**（见下节结论） |
| **B · 视频问答 / 评测** | 离线视觉预算与时间对齐字幕 | **两路评测已完成，长度规则待独立验证**，见本页总设计 |

更多：[`docs/reproduction/README.md`](docs/reproduction/README.md)（3DGS 复现日记）、[`docs/system/README.md`](docs/system/README.md)（系统对接）。全文话术与流程图亦见仓外 [`毕设.md`](../毕设.md) / [`docs/thesis.md`](docs/thesis.md)。

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


## 工作 B：Mage-VL + VLMEvalKit

| 路径 | 说明 |
|------|------|
| [`VLM/mage_vl/`](VLM/mage_vl/) | 基于 microsoft/Mage 的 `mage_vl` + streaming 增量；**无** `data/`/`models/` |
| [`VLM/VLMEvalKit/`](VLM/VLMEvalKit/) | 评测框架 **vendored 完整代码（含本仓改动）** |
| [`VLM/patches/vlmevalkit/`](VLM/patches/vlmevalkit/) | 相对上游 open-compass 的 diff（说明用） |
| [`VLM/eval_results/`](VLM/eval_results/) | **Video-MME 等准确率结果**（必留） |

`mage_flow`（文生图）**不在本仓**，仅存于机器上仓外 `../Mage/mage_flow` 等。  
权重与数据集放仓外或本地 ignore 目录；环境见下表与 [`VLM/README.md`](VLM/README.md)。

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

详情：[`VLM/README.md`](VLM/README.md) · [`docs/PLAN_mage_query_IP_select.md`](docs/PLAN_mage_query_IP_select.md)。

| 同 K · short200 · kr=0.75 | Acc | 备注 |
|------|-----|------|
| **queryip · 主线 A** | **63.5%** | [`clip_frame_n200/`](VLM/eval_results/queryip_samek/clip_frame_n200/) |
| random | 59.5% | 同上 |
| CLIP / region·shuf / edge·shuf | 58.5 / 58.5·60.5 / 60.0·62.0 | 失败 |
| OCR textseat / full（v1 proxy） | 37.3 / 84.1 | [`ocrbench_v1_image_textseat/`](VLM/eval_results/queryip_samek/ocrbench_v1_image_textseat/) |

| 路径 | 说明 |
|------|------|
| [`VLM/README.md`](VLM/README.md) | **应用场景全文** + 实验结论 |
| [`VLM/Explore-Then-Select/`](VLM/Explore-Then-Select/) | Query-IP + [`COMMANDS.md`](VLM/Explore-Then-Select/COMMANDS.md) |
| [`VLM/eval_results/`](VLM/eval_results/) | 指标归档 |
| [`docs/PLAN_mage_query_IP_select.md`](docs/PLAN_mage_query_IP_select.md) | 结论 · 证据 · 后续实验 |

`mage_flow` **不在本仓**。权重 / 大数据集放仓外或 ignore。

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
# 评测时将 editable 路径改为本仓 VLM/VLMEvalKit
```

---

## 布局

见 [`STRUCTURE.txt`](STRUCTURE.txt)。忽略：`.venv*`、`ddgs/`、`_archive/`、`**/data/`、`**/output/` 大产物、权重。

---

## TODO

**工作 A 消融已齐**（2×2 / V2 负对照 / CADR A-full+C4 / P2 classic-repair+garden / P3 iter-sweep+depthft）。详表：[`3DGS/eval_results/`](3DGS/eval_results/)。

**后续可选**

- [ ] 用 [`lineB_kitchen_iter_sweep.csv`](3DGS/eval_results/lineB_kitchen_iter_sweep.csv) 出论文 **PSNR–iter** 图
- [ ] **V2 尺度对齐**（非必须；11.78 负对照即可）
- [ ] 工作 B：Cultural LoRA / codec 选区等（见 [`docs/PLAN_mage_cultural_lora_location.md`](docs/PLAN_mage_cultural_lora_location.md)）

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


### A.I.R.题面条件编译：DeepSeek v2（2026-11-05）

仍属于语义时间证据选择模块。程序固定事件ID、题面顺序、明确范围及before/空间/相对位置关系，DeepSeek仅编译受约束的检索短句；不观察视频、不预测答案。本版只开放旧开发集5题，40项纯CPU测试和实际5题结构核对已通过；真实API/定位/QA尚未运行，不能视为涨点结果。旧Mage解析失败结果与既有A.I.R./字幕结果保留原位。

实施与可续跑入口见 [DeepSeek文本条件编译计划](docs/PLAN_condition_text_parser_deepseek_v2.md)，旧错误见 [Mage解析5题失败诊断](docs/ANALYSIS_condition_grounding_parse5_20261105.md)。新增成果统一保存于 VLM/eval_results/practical_mage_v1/18_air_frames_v1/condition_grounding_deepseek_v2/。先核对5题含义，再决定扩样与定位，不自动进入下一阶段。


### 30题条件解析的离线校验修订（2026-11-05）

已按原文区间区分题目套话、检索事件和问题关系，复用每题最终缓存重审：原23题不变，6题恢复文本检查通过，1题仍不支持拆分；无新增API/视频/QA调用。after与未知未来时间另存结构约束，否定、服饰、动作等仍严格保留。这是校验误拒修复，不是定位或准确率收益。见 [离线重审计划与实际结论](docs/PLAN_condition_text30_offline_reaudit.md)。全部新结果在 VLM/eval_results/practical_mage_v1/18_air_frames_v1/condition_text30_offline_reaudit_v1/，原实验结果保留。


### 查询到画面的独立定位诊断入口（2026-11-05）

已搭建run_leaf_probe.sh，固定7题候选、4题可见锚点探测、19题原回退，复用冻结查询和历史图像特征，只补缺失的18条本地CLIP文本向量。输出原始检索峰、数值时间约束、拟议最多32帧及原帧/PTS审计缩略图。无最终QA或API调用；人工7/4/19范围属于已见开发集诊断，不是通用自动分类或涨点成果。30项CPU回归、模拟全流程与真实来源预检已完成，等待用户运行编码/预览。见 [定位诊断计划与终端命令](docs/PLAN_condition_leaf_localization_probe.md)。新成果位于VLM/eval_results/practical_mage_v1/18_air_frames_v1/condition_leaf_localization_probe_v1/，旧结果不改。


### 冻结7候选QA＋23题原答复用诊断（2026-11-05）

已搭建run_leaf_qa30.sh：先对完整30题重建旧uniform/noobs输入并核对完整Tensor SHA，再复用23题；只对原7候选的新冻结画面生成，484失败案例保留，不改选帧/QA提示词。完整30题报告相对noobs与uniform的救回/误伤/净增，单列prior12/extra18，保留历史收益和历史成本，不能将其归因本次改动。43项CPU回归、模拟中断/全流程与实际来源元数据预检已完成；真实输入审计与QA等待用户运行。见 [冻结7题QA诊断计划及命令](docs/PLAN_leaf_frozen7_qa30.md)。新成果统一在VLM/eval_results/practical_mage_v1/18_air_frames_v1/leaf_frozen7_qa30_v1/，无新增API、训练、下载或自动扩样。


### 冻结7题QA已完成：新增模块无净收益（2026-11-05）

uniform32为16/30；旧noobs与本轮均18/30。相对noobs救回0、误伤0、净增0，7题新输入已重新生成但答案全部相同。extra18仍12/18，其中改输入仅3题。相对uniform的2题净增属于旧noobs，不能归因新增解析/定位。本配置归档，不据此扩全量；7题高基线小样本不能否定所有语义路线。见 [实际结果与边界](docs/ANALYSIS_leaf_frozen7_qa30_outcome_20261105.md)。原结果保留。


### Codec dev180已完成（2026-11-06）

Codec无字幕111/180；完整字幕13312为127/180；32锚点2048为123/180；Frames连续2048为119/180。完整字幕相对Codec净增16题/+8.89pp，开发聚类95%区间[1.67,16.11]；相对Frames和2048的额外收益区间跨零，不宣称稳定领先。固定配置，下一步先预检val90缓存/预算再做补充验证。540单元审计通过，全部数据保留。见[实际结论](docs/ANALYSIS_codec_dev180_complete13312_20261106.md)。
