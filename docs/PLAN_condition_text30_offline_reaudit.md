# 固定30题缓存离线重审：套话、事件内容、问题关系分离（2026-11-05）

状态：独立工程修订；仅重审缓存最终响应，不新增付费请求，不启动定位或QA。修订原因是旧校验把According/Based/following等题目套话及will/we等疑问语法当作必须进入检索短句的内容。原30题结果23题通过、6题拒绝、1题不支持，全部保留原位。

## 1. 本轮处理原则

- 不改旧提示词、旧程序协议、模型响应或原汇总；不选择更有利的早期尝试。每题只取旧逐题记录指向的最后一次响应，并核对缓存SHA、正常stop结束与raw_text逐字一致。
- 规则输入仅为原题、完整选项及冻结程序结构，不接收题号、真值、旧QA预测或模型漏词列表来决定删哪些条件。
- 不是扩大全局停用词。所有豁免都有原文文档、字符区间、原句和类型；同一个词出现在区间外的对象/动作中仍须保留。
- 对象、动作、服饰、颜色、数量、否定及事件ID绑定仍受原词汇校验。给定before边、未知顺序、明确数字范围、首次/末次、开头/结尾等结构不改，不能因此升级任何定位能力。

## 2. 三类内容及修正方式

**题目套话**：仅识别句首完整According to the content of the video, / Based on the video, / Based on the information provided by the video,。which of the following仅在句首或已识别套话之后作为选题操作符，不能作用于标题/引文中的同名短语。来源中的这些词允许出现，但不强制进入检索短句。

**检索事件**：按原事件的来源区间逐词标注。所有非豁免位置的内容词仍必须出现，且不得混用其他事件的词或选项才有的原因。旧schema的源词白名单与“必须保留”集合分开，避免Based/视频等可省词被保留时反而误判为新词。

**问题关系**：What happened after/before X中的X是给定锚点，待回答事件未知。What happened作为疑问操作符，after/before另存unknown_successor/unknown_predecessor、原词区间与锚点引用，不构造虚假的已知事件或SEQ边。When will we只有在原契约为narrated_time_or_schedule时才拆为requested_future_time操作符，未知未来时间仍保存在结构中。不能只删after/will而不记录其含义。

因而406的锚点查询可以不含after，但整体任务仍要求回答锚点之后的内容；本轮没有证明后续窗口覆盖或推理能力已经解决。506的开头/结尾与口述时间依赖也保留。259的isn't mentioned仍是必须保留的否定，且原absence_unimplemented能力限制不变。

## 3. 工程与审计

新增condition_text30_reaudit.py、condition_text30_reaudit_run.py、run_condition_text30_reaudit.sh、test_condition_text30_reaudit.py，原解析脚本不改。

结果独立放在：

`VLM/eval_results/practical_mage_v1/18_air_frames_v1/condition_text30_offline_reaudit_v1/`

candidate.json锁定原清单、全部缓存/逐题记录SHA、新代码和逐题区间划分。records逐题保存离线判定、原错误、最终响应引用、原记录SHA及新规则SHA。results/summary.json和review_packet.json区分unchanged_valid、recovered_cached_final、still_invalid、unsupported，并展示原结构/新划分/原最终查询。日志与工程检查均在新目录。

程序主动禁止网络连接/DNS、真实env、真值、旧QA预测、人工证据、视频/权重读取和视觉库导入。无需DEEPSEEK_API_CONFIRMED，不调用旧parse动作。只读原缓存，原历史API用量单列保留；新增API、视频解码与QA均为0。

27项新增CPU回归，加原52项，共79项。主要检查局部豁免而非全局停用词、同词在非套话位置仍必需、否定/服饰/动作/数量/先后关系不可丢、结构篡改拒绝、事件保序、JSON重复键拒绝，以及只取最后缓存、截断不可被恢复、离线防护阻断网络与密钥。

每次入口核对分支、df与目录大小，空间<10GiB或AIR目录>1GiB停止，无安装/下载/自动清理。逐题可恢复，带tqdm；代码/来源漂移不覆盖旧记录。

## 4. 用户可复现命令（纯CPU）

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_condition_text30_reaudit.sh test-cpu &&
bash VLM/mage_vl/air_v1/run_condition_text30_reaudit.sh init &&
bash VLM/mage_vl/air_v1/run_condition_text30_reaudit.sh run &&
bash VLM/mage_vl/air_v1/run_condition_text30_reaudit.sh summarize
```

## 5. 结论边界与下一阶段

恢复6题若成立，只说明纠正了词汇判定误拒，是事后工程修正，不能写成DeepSeek涨点或方法精度。不能把29题文本检查通过写成29题语义定位成功，更不能把672不支持算成功。

下一步先审阅新packet中可见事件、已知/未知关系及能力限制，再为受支持的叶事件复用图像特征作定位。主旨、口述原因、否定/计数/首次等不因本次修订自动获得定位能力；保留原回退范围。没有定位可行性证据前，不运行最终QA、不叠字幕或Codec，不扩全量。

前序：[30题计划](PLAN_condition_text30.md)、[5题职责修订](PLAN_condition_text_parser_deepseek_v2.md)。


## 实际完成结果

已在Mage环境完成79项CPU回归、实际30题离线重审及续跑一致性检查。原23题结果不变，265/406/389/259/506/516共6题恢复文本检查通过；合计29题文本检查通过，672仍不支持拆分，不算成功。全部使用各题最后一次缓存，不改一个查询字、不挑早期尝试；来源文件哈希不变。新增API/模型生成/视频解码/QA均为0。结果见独立目录results/summary.json与review_packet.json，工程验证见engineering_checks/final_verification.json。

本轮属于事后修复过严校验，不能解释为新算法效果。整体语义/视觉定位仍未认证；下一阶段先保留能力边界，再考虑受支持叶事件的定位。
