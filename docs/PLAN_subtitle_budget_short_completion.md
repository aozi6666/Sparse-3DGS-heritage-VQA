# 连续字幕基线收尾：3072 预算对照与 short219

## 当前结论（2026-11-03，字幕路线已完成冻结评测）

保留Frames-32＋连续补句＋时间标记，固定2048字幕token；无训练、无Agent/API。test225三路正确数：无字幕140、原对齐154、连续163（72.44%）。相对无字幕净增23题／10.22个百分点，聚类95%区间[5.33,15.11]；相对原对齐净增9题／4个百分点，区间[-0.44,8.46]，尚不能证明补句独立显著优于原对齐。

short219/dev180/val90/test225连续字幕分别175/119/54/163题正确，四阶段互斥覆盖本地714题／238视频，不能混称714题独立测试或官方完整基准。原对齐val90负结果、3072预算负结果及API pilot负结果全部保留。当前不再调字幕或重复跑本地full；Agent暂停，文旅闭环和重建对问答的贡献尚待验证。

完整三路数据、分时长统计、5组图、成本和10个案例见[最新最终结果总结](SUMMARY_mage_subtitle_final_results_20261103.md)。历史内容若与本节不符，以本节及最新总结为准；历史实验不删除。

---

## 以下为此前阶段记录／原实施说明（历史保留）

日期：2026-11-03。工作分支：`vlm`。

本轮先完成字幕基线，不开发 Agent，不调用 DeepSeek，不动 test225。所有推理均由用户在终端手动启动。

## 1. 当前证据和本轮问题

原 dev180（60 个视频）的 Frames-32 无字幕、原对齐字幕、连续补句＋时间标记 2048 分别答对 **110、117、119/180**。连续字幕相对无字幕救回 20、误伤 11，净增 9 题；相对原对齐字幕只净增 2 题。总体置信区间仍跨零，长视频没有净收益，不能写成稳定或显著提升。

2048 配置省略了 8 个视频的部分邻句。这些视频完整邻句最多需要 2649 个文本 token，因此只增加一个固定 **3072** 对照，不继续扫描预算。3072 检验的是邻句被预算省略的影响，不能补齐远离全局采样帧的证据。

“原对齐字幕”保留原来帧命中、去重、按时间排列的输入；“连续字幕”在命中字幕附近按统一规则补相邻句并加时间标记。二者必须分别报告，不能改写旧实验。预算是字幕文本块的 tokenizer 计数，**不是视觉＋问题＋回答的总上下文上限**；新增文本成本要单独报告。

## 2. 新代码与固定范围

入口：`VQA/mage_vl/practical/run_subtitle_completion.sh`。

- `subtitle_completion_common.py`：锁定旧协议、字幕源、证据和配置，逐条审计结果及复用来源。
- `subtitle_completion_run.py`：原生输入校验、限定范围推理、逐答案落盘和 tqdm。
- `subtitle_completion_analyze.py`：完整后才读标签；配对统计、分时长结果、成本及归档；显式冻结 short 配置。
- `test_subtitle_completion.py`：22 项新增 CPU／模拟检查；与已有 111 项检查共同通过。

复用既有原生 Mage 模型、32 帧采样、150000 像素上限、3136 下限、确定性生成和最多 32 个回答 token；不训练、不下载、不复制模型、视频或帧。运行环境、模型和数据身份必须与旧实验一致；漂移时停止并保留现场。

## 3. 第一步：只做 3072 对照

开发集仍为原来的 180 题，不按答案选择样本：

- **8 个视频／24 题**：3072 的实际字幕输入改变，仅这些题允许新生成。
- **其余 156 题**：重新构造旧 2048 与新 3072 输入，比较每个 tensor，并核对旧日志保存的输入 token 哈希、原文、视觉布局、视频与协议身份；全部通过才复用旧回答。不能把“预算参数变了但文字似乎一样”当成复用证据。
- 旧日志没有保存历史像素 tensor。本轮验证的是同一已锁定来源重新构造的 2048／3072 tensor 相等，并与旧 token 哈希及视觉记录一致，不能声称直接比较了历史像素缓存。
- tqdm 的 **180 个答案**包括上述复用检查，**不是 180 次新推理**。每个完成单元立即封存、刷新落盘，重复执行自动跳过。出现输入不符，不擅自把复用范围扩大为新推理。

同时报告完整 180、变化 24、复用 156、中等／长视频上的正确数、救回、误伤、净增、视频聚类 bootstrap 区间，以及逻辑文本／视觉 token、新增生成成本。旧 2048 文本总量为 **234552 token**，保留在原汇总；本轮证据审计另存两种预算的文本量对照。包含额外原生校验的耗时不能用作公平部署速度结论。

预算决策：先看完整结果，再只固定一个全局配置。若放宽后没有可解释的收益，保留成本更低的 2048；若有正向证据，结合救回、误伤及成本讨论是否采用 3072。不得逐题选择预算，不根据 short 的得分反选配置，也不把开发集结果解释为独立测试。

### 现在执行的命令

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
{ [ "$(git branch --show-current)" = vlm ] || git switch vlm; } &&
test "$(git branch --show-current)" = vlm &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VQA/mage_vl/practical/run_subtitle_completion.sh init budget &&
bash VQA/mage_vl/practical/run_subtitle_completion.sh run budget &&
bash VQA/mage_vl/practical/run_subtitle_completion.sh summarize budget
```

允许中断后原样重跑。`git switch` 不带任何强制选项；如果现有改动导致切换被拒绝，停止，不覆盖或丢弃改动。

## 4. 第二步：固定预算后补齐 short219

short219 的 73 个视频字幕源已做 CPU 清单审计：**56 个有 SRT、17 个缺字幕**。不下载补字幕，也不删除缺字幕样本；缺失、空白或没有时间命中的视频，字幕路线回退同题无字幕回答，全部保留在 219 题分母。实际帧命中状态在用户运行时确定。

固定配置后，对全部 219 题比较三路：Frames-32 无字幕、原对齐字幕、连续补句＋时间标记。三路视觉输入一致；字幕 token 和总输入 token 单列。657 个逻辑答案分别断点保存。原 6 题冒烟记录只有实际输入身份检查通过才复用；历史 short219 输出不直接作为当前三路对照，以免混入旧提示词、流程或身份不足的记录。还会报告冒烟之外 213 题。

以下命令**等预算汇总检查后再用**。其中 `--tokens` 必须先确定为 2048 或 3072；配置文件只允许冻结一次，不自动挑选。本轮没有替用户冻结预算或开始 short。

```bash
# 示例为选择 2048；若预算对照后决定采用 3072，只在首次冻结前替换此数。
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = vlm &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VQA/mage_vl/practical/run_subtitle_completion.sh prepare-short &&
bash VQA/mage_vl/practical/run_subtitle_completion.sh freeze-short budget --tokens 2048 &&
bash VQA/mage_vl/practical/run_subtitle_completion.sh init short &&
bash VQA/mage_vl/practical/run_subtitle_completion.sh run short &&
bash VQA/mage_vl/practical/run_subtitle_completion.sh summarize short
```

short219 已参与历史开发，只能作为覆盖性评测。val90 已参与诊断。**test225 继续保留**；本入口不提供 val、test、full 模式。

## 5. 保存与省盘

全部新产物归 `VQA/eval_results/practical_mage_v1/`：

- `14_subtitle_completion/budget3072_candidate.json`：预算协议、变化题清单、旧来源和源码哈希。
- `14_subtitle_completion/budget3072_dev180/`：运行清单、逐题预测、汇总。
- `14_subtitle_completion/short219_sources.json`：短视频字幕源身份和缺失状态。
- `14_subtitle_completion/selected_short_config.json`：预算确认后才创建。
- `14_subtitle_completion/short219_t2048/` 或 `short219_t3072/`：冻结后的三路配对成果。
- `14_subtitle_completion/logs/`：每次推理的完整标准输出、错误输出；失败日志也保留。
- `14_subtitle_completion/evidence_budget_audit_v1.json`、`cpu_checks_v1.json`：预算文字审计和静态检查记录。
- `cache/subtitle_completion/<配置签名>/`：逐答案 JSONL 检查点、证据、输入身份、来源及环境；不是模型权重检查点。
- `07_archive/subtitle_completion_budget_t3072_v1.json`、`subtitle_completion_short_t<预算>_v1.json`：汇总成功后封存报告、逐题预测和来源哈希。

代码准备时可用空间约 **34G**，原实验目录约 **151M**。每次入口检查 `df`／相关目录体积，推理中继续检查空间；默认至少保留 10 GiB 空闲，整个 `practical_mage_v1` 目录体积受 1 GiB 预算保护。只新增文本、日志和轻量证据索引，不缓存帧、tensor 或视频，不删除既有缓存。1 GiB 是保护上限，不是预期用量；长时间重复重启产生的日志也计入目录空间检查。

## 6. 何时再做 Agent

先拿到预算对照和 short219 结果，再更新“短／中／长 × 三种字幕配置”的对照结论，确定连续字幕基线。之后才另立主动取证实验，处理远离采样点的证据缺失与错误覆盖；不得把本轮预算对照当成已解决长视频问题。
