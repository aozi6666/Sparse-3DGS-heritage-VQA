# 固定30题条件编译扩展（2026-11-05）

当前阶段：补齐原开发30题的结构并审阅文本；不开放定位或最终QA。沿用已通过5题的DeepSeek提示词、模型、生成上限、JSON契约和词汇校验，不叠字幕、Codec或新判决。

## 输入与成果保护

30题严格取condition_clips_noobs_qa30/candidate.json已有indices；仅使用原题与全部选项。不读取GT、旧预测、人工证据范围，不引入新数据集。旧5题的程序契约和响应逐字复用，核对父协议及记录SHA；不再次付费。新结果独立放在：

`VLM/eval_results/practical_mage_v1/18_air_frames_v1/condition_grounding_deepseek_v2_parse30/`

candidate.json锁定源码/输入/父记录。parses、responses、attempts.jsonl、logs及parse_review/parse30中的summary/review_packet均留存。原parse5与所有历史实验不改写。新报告单列旧5题调用成本，避免复用被误写成部署零成本。

## 程序结构和明确边界

显式事件沿用原ID，已知before边与未知顺序分开；数字范围来自题面。同场排列提取全部选项共有实体，不继承某一选项的顺序。

新增题按通用题面规则区分全局概括/比较、计数、首次/末次/第几案例、未知前后事件、触发事件与待推断结果、比分条件、口述时间、否定。限定词另存原文，不把比分38-29、第二案例、首次魔术等强行换算成绝对秒数。why和机制问题保留原因未知标记，文字/对白/全局覆盖依赖单列。

不能确定性拆分的题标unsupported，不伪造事件身份，不算成功解析。例如672只在选项列举顺序，walking与taking a walk等用词不完全相同；本轮不做未经审阅的语义合并，也不选一个选项当真值。该题不调用API，后续暂精确回退既有noobs输入。其余未实现的计数/全局/否定等能力也只标记边界，文本编译不代表能力已实现。

报告覆盖全部30题，区分旧5题复用、有效编译、API失败、程序不支持，不能只挑成功题。严格保留内容词可能保留冗余表达；先看原始输出，不以放松词汇校验制造通过。

## 实现与检查

新增condition_text30.py、condition_text30_run.py、run_condition_text30.sh与test_condition_text30.py，复用旧condition_text_parser/API而不修改它们。标准库+tqdm，无新增依赖、下载、视频抽帧或视觉模型加载。52项CPU回归包含原40项及新增12项类型/未知事实保护。

新阶段累计API请求最多50次（含失败/不确定请求）；记录token停止线60000，并非金额硬上限。预计旧5题复用、1题unsupported、24题新增请求；每题至多一次格式修复。请求成功先保存响应再解析，重启复用缓存；不确定网络请求默认停下，人工核查后才可加--retry-uncertain。仍要求剩余空间10GiB、AIR目录不超过1GiB；禁止自动清理。密钥由程序读取已有.env/deepseek.env，不输出、不进入协议。

## 用户本次执行

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_condition_text30.sh init &&
bash VLM/mage_vl/air_v1/run_condition_text30.sh preflight &&
export DEEPSEEK_API_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_condition_text30.sh parse30 &&
bash VLM/mage_vl/air_v1/run_condition_text30.sh summarize30
```

可重用原命令继续未完成题，带tqdm；旧5题与unsupported题直接计入已有记录。输出格式通过仍须逐题审阅，不能据此宣称定位有效或涨点。

## 后续顺序（本次入口不执行）

1. 审阅完整30题，包括unsupported与非视觉限制。核对目标/动作/服饰、事件身份、已知关系和未知关系；不根据GT调查询。
2. 条件正确后才接CLIP叶事件定位，复用已有1fps图像特征，仅新增必要文本特征。保持全局8帧+6个12秒片段、最多32帧及原预算；保留真实时间轴。按关系能力决定受支持定位与精确回退，不把窗口分配当作证据充分。
3. 检查定位可行性后才开放最终QA。比较均匀、noobs和条件检索，重点报告新增18题救回/误伤/净增；保持原QA提示词，不把解析文本输入答案模型。
4. 若定位无改善或证据到位仍错，停止堆规则；保存失败结果。证明新增题收益之前，不接字幕/Codec，不扩全量。

前序：[5题实现计划](PLAN_condition_text_parser_deepseek_v2.md)。当前是已见开发集可行性研究，不是独立测试。
