# HiMu启发的最小条件解析与子事件定位 v1

先验证条件解析，再定位叶事件，最后才运行QA。当前只完成工程与CPU验收；没有生成解析、定位或新答案，不承诺涨点。旧候选全部保留。

## 本版做什么

纯文本Mage只看原题/完整选项，输出证据类型、绑定目标/动作、原题引用、关系类型、明示数字时间范围和缺口。没有视频、GT、人工位置或旧答案输入。先5题：543时间范围、625未知顺序、484空间排列、549已知before/可能对话、325人物动作绑定。每题最多512新token，结构失败只修复一次，原文留档，不无限重试。

解析通过不等于语义正确。例：first six minutes应为0–360秒；first magic不能硬编码0–24秒；未知顺序不得提前选某个排列。暂不支持以纯CLIP证明first/final/count/absence；speech/mixed/unknown也回退旧noobs输入，保留全部样本与原因。这是检索能力回退，不是根据答案或GT选择路径。

受支持视觉题中，每个绑定的目标＋动作单独编码CLIP，复用旧1fps图像特征。原完整题目/选项用于解析和最终QA，旧完整query路线作为冻结回退，不额外设置混合权重。已知先后关系采用HiMu启发的严格过去/未来门控；未知顺序、空间排列不强加时间顺序。优先给每个子事件分到窗口，再轮选补足6窗口。分数不是事实概率，分到窗口不代表定位正确。

保留8全局帧＋6个12秒窗口、每窗口4帧，总上限32，与旧协议一致；重复端点/锚点可能少于32，不额外补帧改变预算。数字时间范围约束局部窗口；全局锚点仍覆盖全片。范围或子事件约束无法满足时精确回退旧输入，不能放宽或按答案补窗。

## 先执行：5题纯文本解析

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_condition_grounding.sh init &&
bash VLM/mage_vl/air_v1/run_condition_grounding.sh preflight &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_condition_grounding.sh parse5 &&
bash VLM/mage_vl/air_v1/run_condition_grounding.sh summarize5
```

只有5–10次纯文本调用，不解码视频、不生成视频QA、不调用CLIP或API。首先查看parse_review/parse5/review_packet.json，确认条件与叶事件没有被错误改写。schema失败会保留报告并停止，不将非法输出伪造为中性结果。若失败需诊断新版本，不能覆盖旧断点。

## 审阅后再开放的阶段（不要直接串成自动全跑）

原5题语义可用后，设置PRACTICAL_PARSE_REVIEWED=1，运行parse30与summarize30，前5题复用。审阅30题的事件引用、能力缺口与回退分布后，运行ground、ground-report。定位阶段只生成新增叶文本向量，复用旧图像特征，未运行最终QA。

```bash
# 在项目且已激活环境；仅在真实审阅解析后设置，结构合法不能代替语义认可。
export PRACTICAL_STAGE_CONFIRMED=1 PRACTICAL_PARSE_REVIEWED=1
bash VLM/mage_vl/air_v1/run_condition_grounding.sh parse30 &&
bash VLM/mage_vl/air_v1/run_condition_grounding.sh summarize30
# 继续前先审阅parse30/review_packet.json
bash VLM/mage_vl/air_v1/run_condition_grounding.sh ground &&
bash VLM/mage_vl/air_v1/run_condition_grounding.sh ground-report
```

定位审阅必须对照原视频，检查峰值是否真是目标事件、所有必要事件是否找到；只看CLIP分数或窗口分配表不够。公开KFS只能离线辅助诊断，不能给自动定位提供窗口。若仍错事件就暂停，不叠新规则；本版没有假定模型自述事实可信。

只有叶定位值得继续，才设置PRACTICAL_LOCALIZATION_REVIEWED=1后运行qa、summarize-qa。不变输入复用noobs的完整输入tensor身份与答案；变动输入沿用已验证的原生回归与QA代码，只生成最终答案，不把解析文字输入Mage。

```bash
export PRACTICAL_STAGE_CONFIRMED=1 PRACTICAL_LOCALIZATION_REVIEWED=1
bash VLM/mage_vl/air_v1/run_condition_grounding.sh qa &&
bash VLM/mage_vl/air_v1/run_condition_grounding.sh summarize-qa
```

四路统计：均匀、带观察v2、去观察noobs、本版。分别报告中/长、旧12/新增18；所有unsupported样本保留。开发30题不是独立测试，旧12改善不能替代新增题收益。只有定位和新增题结果支持，才讨论扩展，当前无自动full/val/test入口。

## 结果、恢复与成本

新根：VLM/eval_results/practical_mage_v1/18_air_frames_v1/condition_grounding_v1/。

- candidate.json：原始题目、源码、模型和输入身份；没有GT。
- parse_calls/、parses/：逐次原始解析与断点，含修复尝试和成本。
- parse_review/parse5/与parse_review/parse30/：summary.json、review_packet.json。
- leaf_features/、grounding/：逐题叶查询向量、峰值、窗口、回退、是否改变输入。
- ground_review/：定位审阅包，semantic_grounding_pass保持未认证。
- qa/results/：最终四路对照汇总、逐题预测、断点归档。
- logs/、engineering_checks/：终端日志、CPU验收及隔离记录。

每阶段重复同一命令可恢复。所有文件与来源SHA校验；冲突、损坏或残行停止，不自动删改。新增解析和文本编码成本单独记录，不能只报告最终QA耗时；原图像特征历史编码仍有成本。沿用10GiB空闲与整个AIR根1GiB保护，缓存均在新根，不保存原始帧、不复制权重、不安装下载。无Codec、字幕、ASR、OCR、API或新训练。

## 来源与验证边界

借鉴HiMu的子谓词与已知时序组合思想，未移植其LLM/多专家/完整PASS管线，不是官方复现。源码归属见LICENSE.HiMu。参考：https://github.com/DanBenAmi/HiMu ，本地研究审计见docs/RESEARCH_semantic_evidence_next_20261105.md。

已通过34项CPU测试（30项新契约＋4项原生处理器测试），5题真实纯文本tokenizer输入保留全部选项、没有视觉占位；人工定位/GT/旧答案文件读取拦截未触发。没有加载模型权重或生成解析。Mage纯文本forward分支已静态核对，但其在本任务上的实际解析质量必须由parse5验证。
