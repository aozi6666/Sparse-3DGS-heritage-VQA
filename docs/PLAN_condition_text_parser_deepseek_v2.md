# A.I.R.条件解析职责修订：DeepSeek受约束查询编译（2026-11-05）

状态：新独立版本，仅开放旧开发集固定5题文本检查；尚无真实API输出、定位或QA效果。旧Mage解析v1、A.I.R.、noobs、字幕及全部缓存原位保留。

## 1. 模块边界与本轮目的

本模块仍属于A.I.R.启发的语义时间证据选择，位于“原问题→条件/子事件→CLIP检索”之间。DeepSeek仅处理题面，不看视频，不充当答案裁判；最终答案仍由原Mage QA产生，但本轮入口尚不开放QA。

HiMu启发来自“拆分叶事件、区分已知关系和待推断顺序、再分别定位”。本版只做受约束的查询编译，不是HiMu完整复现、官方A.I.R.复现，也没有新的特征融合层。Codec与字幕均不在本轮输入中，后续组合收益不能由独立模块相加推算。

旧v1解析5题仅2题格式合法，而这2题也有事件重排、丢弃已知before条件等问题。本版检验职责拆分是否能避免这些错误；不是准确率实验，不预先承诺涨点。

## 2. 程序持有的结构

- 明确数字范围由程序解析并保留原文字区间：例如first six minutes为[0,360]秒，不视作first_event。
- 题面(a)…(d)事件沿用原ID和列举顺序；列举顺序不是视频中的已知时间顺序，不构造SEQ边。
- 对“单一画面内四作品排列”，仅当四组选项包含完全相同的四个引号实体时提取共有集合，使用稳定字典序编号；不采用某个选项的排列，也不预设空间顺序。
- 题面给定的Before/After保留为有向边，事件ID仍按原文位置编号。例如“Before A, B”记录B早于A，不把源事件列表改成候选答案顺序。
- “采访在视频哪一部分”记录relative_position，不设置未知多事件顺序或伪造绝对时间范围。
- 每个事件保存来源文档、字符区间与原文；对象、动作、服饰、颜色、数字等内容词是查询不可丢失的条件。
- why/对白/文字依赖只作限制标记。定位人物坐姿或发短信不等于已确定动机；引用存在、词汇保留也不等于语义成立。

这是保守、有限覆盖的程序规则，不宣称通用自然语言理解。当前入口只开放固定5题。其余25题可能需要更多题面类型规则；先看5题实际输出，不因为JSON合法自动扩到30题。

## 3. DeepSeek只做什么

发送原问题、完整ABCD选项、程序固定的事件/范围/关系。选项明确标为未验证的备选，不当成已发生事实。模型唯一允许的输出是：

```json
{"queries":[{"event_id":"固定ID","text":"仅根据该事件原文编译的短句"}]}
```

模型不得修改ID、事件顺序、关系或时间，也没有答案、事实、置信度字段。每条最多48词/260字符；允许常规词形变化，拒绝引入其他事件/选项才有的内容词，拒绝丢弃来源内容词。此严格约束刻意偏向“少改写、保条件”，而非自由生成精巧但失真的查询。

校验仅能检查结构和词汇。动作方向、修饰关系等语义仍须审阅；全部格式合法也不能写成“条件理解成功”或“选帧有效”。如果原文直用与LLM改写差异很小，应如实保留程序解析贡献，不将API本身包装为性能来源。

## 4. 独立实现与冻结配置

代码位于VLM/mage_vl/air_v1：condition_text_parser.py（程序契约与查询检查）、condition_text_api.py（标准库HTTP/凭据加载/请求断点）、condition_text_run.py（固定5题入口）、run_condition_text_parser.sh，以及两份CPU测试。

结果根目录：`VLM/eval_results/practical_mage_v1/18_air_frames_v1/condition_grounding_deepseek_v2/`。

- candidate.json：固定5题、全部原始输入、结构、源码/来源SHA、API配置；不包含密钥。
- responses/<题号>_<尝试>.json：发送的无密钥正文、原始响应、用量、耗时、请求/记录SHA。
- parses/<题号>.json：逐题最后结果、失败原因和尝试来源；失败也保留。
- attempts.jsonl：持久请求计数；日志残行/哈希损坏直接停止，不自动丢弃。
- parse_review/parse5/summary.json、review_packet.json：计数、结构/查询对照、语义审阅pending。
- logs、engineering_checks：运行日志与离线工程检查。

模型deepseek-flash，官方HTTPS端点，thinking disabled，temperature0，JSON输出，max_tokens768。每题最多2次（初次+一次受限格式修复）；不是加长Mage JSON，也不是多轮答案验证。持久化请求上限10次（含失败/不确定请求），记录token停止线30000；后者不是金额硬上限。托管模型别名/temperature0不能冻结服务端权重或保证输出完全一致；记录服务端model字段及时间。

仅用Python标准库和环境中已有tqdm，无安装、下载、视频解码、GPU模型加载。新增内容主要是几份文本JSON；复用旧题面清单，不复制视频、权重、CLIP图像特征。入口每次检查分支、df和目录大小，空间<10GiB或AIR目录>1GiB即停，不清理已有文件。

凭据由Python在实际请求前加载，不source shell，不输出密钥。支持根目录.env或deepseek.env；两者同时存在则用DEEPSEEK_ENV_FILE指定。已export的DEEPSEEK_API_KEY优先。只读取API_KEY这一项，不执行变量/命令替换。助手仅检查路径，测试使用临时假凭据；不查看真实env内容。

## 5. 可续跑与失败处理

已保存响应先核对请求SHA再复用；即使在原响应保存后、逐题单元完成前中断，也能继续编译而不再次付费。已完成题（包括2次后仍失败的题）不重采样挑结果。代码/来源/配置漂移拒绝覆盖，修订需另立版本。

若请求发出后网络中断且无完整响应，程序无法确认服务端是否已计费，默认停止。人工核查后可显式向parse5追加`--retry-uncertain`；重发计入原有10次限额。HTTP失败也不自动重发。不在终端或日志打印Authorization或HTTP错误正文。

## 6. 用户执行到5题为止

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_condition_text_parser.sh test-cpu &&
bash VLM/mage_vl/air_v1/run_condition_text_parser.sh init &&
bash VLM/mage_vl/air_v1/run_condition_text_parser.sh preflight &&
export DEEPSEEK_API_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_condition_text_parser.sh parse5 &&
bash VLM/mage_vl/air_v1/run_condition_text_parser.sh summarize5
```

当前仅deepseek.env存在时自动使用，无须再次复制密钥。正常中断后重用原命令；不删除旧缓存。脚本不会自动进入30题/定位/QA。

## 7. 之后的阶段判据（本次不执行）

1. 逐条审阅5题短句：543保留发短信与数字范围、不得生成选项原因；625保留a/b/c/d四事件且无已知时序；484四作品完整、未知空间排列；549服饰/坐姿和已知before边完整、原因仍未知；325目标+采访绑定并保留相对位置任务。
2. 若仍失败，先诊断程序槽位/词形检查或API短句，不放宽条件以制造通过；不增加token/轮数来无限重试。
3. 审阅后再为固定30题补齐确定结构；只在条件身份正确后分别做CLIP叶事件定位。继续使用既有图像特征，先看定位；候选窗口分配不等于正确证据。
4. 定位可行后才用固定≤32帧、原QA和不变预算做受限对照：原生均匀、noobs候选、新条件查询。重点看新增18题救回/误伤/净增，不能在旧12题诊断集上反复挑规则。
5. 若叶事件定位仍不准，或证据到位后Mage仍错，及时停止该路线；非视觉事实转交字幕路线单独报告。证明新增题收益前不接Codec/字幕或新答案裁判。

## 8. 参考与准确归因

- [HiMu官方代码](https://github.com/DanBenAmi/HiMu)：子事件逻辑/定位思想，未知顺序不得当作已知SEQ；本版未复现其多专用模型系统。
- [DeepSeek官方API](https://api-docs.deepseek.com/api/create-chat-completion/)：本次核对支持deepseek-flash、thinking disabled、json_object；官方亦说明空内容/length截断仍须检查。
- [旧parse5失败归档](ANALYSIS_condition_grounding_parse5_20261105.md)、[此前方案](PLAN_condition_grounding_v1.md)、[语义选帧成果归档](SUMMARY_air_semantic_frames_archive_20261105.md)。
