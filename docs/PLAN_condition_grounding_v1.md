# HiMu启发的最小条件解析＋子事件定位 v1

> 当前状态：2026-11-05 parse5未通过；暂停后续定位与QA。详见[失败诊断](ANALYSIS_condition_grounding_parse5_20261105.md)。以下实施设计与CPU验收为历史记录，不能当实际解析已成功。


2026-11-05，feat/air-visual-evidence。全部旧实验冻结保留。只验证问题条件化检索；不接Codec/字幕/API/判决器，不训练/下载。新成果在18_air_frames_v1/condition_grounding_v1/。用户运行GPU，助手只CPU检查。

## 阶段与边界

先在固定5题（543明确前六分钟、625未知顺序、484空间排列、549已知before且可能需对话、325人物动作绑定）测试纯文本解析；解析器只见完整原题/四选项，不见视频、人工位置、标签或旧答案。用现有Mage纯文本能力，不新增模型/API。JSON最多512新token，结构失败最多一次修复，原始输出保存；失败停止，不强行改写成通过。

结构：evidence=visual/mixed/speech/unknown；events最多4个，每个含target、action、question_quote；relation=independent/known_sequence/unknown_order/spatial_order/first_event/final_event/count/absence；time_range及time_quote用于题目明示数字范围；unresolved保留能力缺口。明确时间范围由原题独立校验；first six minutes必须是0–360秒，不能据first就强制前24秒。未知顺序不得当作known_sequence；空间顺序不当视频顺序；已知关系事件可按问题给出的真实先后排序。事件引用只能来自原题，不能把某个选项当作已知事实。schema通过不证明解析语义正确，必须审阅原始解析包。

5题结构检查通过后才开放同一30题解析。定位阶段另设用户明确阶段确认，不能把形式校验当人工语义认可。所有30题保留，不以真实答案筛选题目。

## 检索最小适配

视觉且关系支持（independent/known_sequence/unknown_order/spatial_order）的有效解析才改变检索；speech/mixed/unknown或first/final/count/absence先精确回退冻结noobs候选并记录原因，不宣称已攻克这些操作。明确时间范围可用于independent等受支持关系。不是答案覆盖门控，最终只走一份视觉输入。

完整原问题/ABCD和原生QA提示保持不变。新叶查询由绑定的target+action组成，单独编码CLIP；复用旧1fps图像特征；原完整query路线保留为旧候选回退，不另设未经验证的混合权重。未知顺序/空间排列分别定位所有事件，不预设答案顺序；known_sequence借HiMu的严格前后门控，分数只是相对检索分数，不是事实概率。

固定8全局帧＋6个12秒不重叠窗口、每窗口4帧，总上限32，与旧协议一致，不额外补到32以改变预算。优先给每个子事件定位窗口，同一最佳时间可共享窗口；余下轮选补足。已知时间范围限制局部窗口，锚点仍全局。无法满足六窗口/子事件约束则显式回退，不放宽预算或按答案修窗。首次/最终等原理上需要更完整证据，本版不以CLIP最早峰冒充解决。

先看叶查询、峰值时间、窗口与回退原因；公开KFS只供离线诊断，不能进入自动定位。若叶事件仍落到错误事件，暂停，不叠新规则。定位可用后才运行最终QA；输入不变复用noobs完整输入身份与答案，其余只新增最终QA。报告四路均匀/v2/noobs/本版，重点extra18，开发30题不是独立测试。

## 可恢复与省盘

解析逐调用、文本向量、定位、QA逐题断点分开SHA锁定；tqdm、GPU共享锁、10GiB空闲与AIR根1GiB保护；不保存帧或复制权重。所有原始失败与回退保存。只有总结阶段读取标签；不会自动扩样或重跑旧480次证据调用。

此为HiMu启发的受限适配，不是官方多专家复现。正式实施顺序：CPU测试与输入隔离→用户parse5→审阅解析→parse30→用户确认定位→审阅叶定位→用户确认最终QA。当前不承诺涨点。

## 工程状态

独立入口run_condition_grounding.sh；完整说明README_condition_grounding.md。34项CPU契约检查通过；5题实际tokenizer输入为纯文本且保留全部原选项，输入隔离拦截无人工位置/GT/旧答案读取。未加载模型权重，未生成解析或QA。纯文本输入只验证接口，不能作为解析效果。

先交用户parse5命令，等实际输出再决定后续；不把parse30/ground/qa串成无人审阅的全跑。分阶段确认源自本任务“先检查解析→检查定位→再QA”的要求，不是额外发布审批。未支持的首次/最终/计数/否定等保留回退，不写成已经解决。
