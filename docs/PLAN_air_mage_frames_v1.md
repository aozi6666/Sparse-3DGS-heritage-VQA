# A.I.R. × Mage Frames：独立选帧工程与分阶段验收

日期：2026-11-03（Asia/Shanghai）；工作分支：`feat/air-visual-evidence`。本文件先于实现写入，并作为中断／上下文压缩后的接续依据。

**当前有效版本：native_uniform_fix1；以第10节修订记录为准。第8节首版67项检查及第9节旧输出路径保留作历史记录，用户命令仍适用。修正版结果为`18_air_frames_v1/native_uniform_fix1/smoke3/summary.json`。**

## 1. 任务边界与成功标准

目标是先跑通可复现的A.I.R.选帧适配，再观察相同模型与最终视觉预算上限下的准确率。接受持平，不承诺涨点，不要求算法创新。只使用现有公开Video-MME本地视频、问题和既有划分，不制作或引入新数据集。

本轮只替换Frames路线的选帧。原生Codec不改、不运行两路融合。原连续字幕2048及全部历史结果保留；首轮不带字幕，避免把新增文本收益误当选帧收益。暂不接DeepSeek、Agent、Query-IP、空间Patch裁剪或新训练。

## 2. 理论依据和参考实现审计

A.I.R.论文：<https://arxiv.org/html/2510.04428v3>，官方代码：<https://github.com/UCF-AIR/A.I.R.>。本地参考commit：`400fb52f1cfbf64debc6408cd215cfa375fd9797`，MIT。原理是粗相似度搜索、VLM真实画面相关性分析、局部时间邻域扩展和预算停止。相关性是寻找证据的代理指标，不是答案正确性保证；计数和时间关系仍可能需要连续过程。

原实现不能直接运行：候选去重后按时间排序破坏潜力顺序；部分分支返回4项而调用方取5项；预算函数与论文自适应公式不一致；Mage提示词使用原Video-MME首尾裁剪会删除选项D；CLIP权重加载失败允许回退未训练权重；旧缓存没有完整模型／来源身份且存在自动删除路径。本工程独立实现有界控制流，逐项保留差异说明，不宣称逐字复现原作者成绩。

资源适配：使用本地HF CLIP ViT-L/14权重，替代论文默认EVA-CLIP；Mage同时担任分析与最终回答模型。没有新增7B模型或下载权重。这里是明确标注的A.I.R.-Mage适配版本，不把结果称为原论文官方复现。CLIP仅作粗筛，不能替换VLM分析与迭代。

## 3. 冻结首版方法

- 全视频1fps粗采样，保存原帧号；CLIP视觉特征按视频可复用，问题文本单独编码。
- 两成分GMM阈值、事件粗候选、区间潜力排序、VLM评分1–5（>3为正）、局部时间补采样。
- 首版最终帧上限32，每轮最多12个新评分帧，最多6轮；严格去重、越界检查、时间排序，保留全部选择轨迹。实际输入可少于32帧，必须报告帧数与token，不能称作严格固定32帧。
- 无有效正证据时按已登记规则回退均匀32，记录fallback，不按真值选路。评分解析失败为invalid，不伪造成3分。
- 最终回答保持原Mage四选一提示词、max_pixels=150000、min_pixels=3136、greedy、seed=0、max_new_tokens=32。分析提示词独立、短回答、评分先行；分析总成本单列。
- 本地真实CLIP tokenizer审计：180题中38题超过75个内容token，最长159个内容token（加首尾后161）。完整问题＋ABCD按最多75个内容token分块，每块加真实BOS/EOS；块向量分别归一化、均值、再归一化。不截断，保存完整token与分块身份；这是本适配的差异，不是原论文新增贡献。VLM始终拿到完整问题和四个选项，不复用原仓首尾裁剪函数。

## 4. 最重要的输入验收

原Mage接收图片列表会把帧号和时间重置为0,1,…，不能直接这样喂非均匀帧。独立适配器复用原生预处理、双重resize约定、原视频帧号构建patch_positions及真实采样时间；不改权重目录下任何代码。

用户第一次运行时，先对均匀32帧进行原生路径与新适配器逐tensor比较，包含pixel_values、grid、patch_positions、input_ids、attention_mask；完全相同才允许继续。每个待处理视频均做输入验收。不额外生成一次基线答案。

## 5. 数据与阶段

只读旧的无真值`01_manifests/inputs.json`、`splits.json`。首轮工程smoke为既有dev30池按原顺序第一个视频的3个问题；扩展dev30为已有开发池90题／30视频，不混入short、val或test。既有714题均已有历史结果，因此一律披露开发／覆盖性质，不把test225重新包装成未见测试。当前入口不开放val、test或full。

先验收CPU逻辑、身份锁和断点，再由用户运行smoke。汇总通过且无输入／缓存异常后，才考虑dev30。第一次仅给smoke命令，不自动扩跑。

## 6. 断点、日志、空间和隐私

新代码：`VQA/mage_vl/air_v1/`，不导入mage_query_ip。
新成果：`VQA/eval_results/practical_mage_v1/18_air_frames_v1/`。内含候选协议、CPU检查、阶段清单、logs、按签名缓存、逐问题选择轨迹、逐帧评分、逐答案记录、summary及归档。全部新持久产物（包括第三方动态代码缓存）留在指定结果根。

评分按问题＋原始帧号存储并立即fsync；普通中断后无需重新付出已完成的评分成本。CLIP每视频特征只存轻量向量与原始索引，锁定视频stat／模型代码、配置、权重stat和问题／提示词哈希。禁止删除旧缓存；失配停止，不静默重建。半行/损坏记录保留现场并停止，不截断。

同一GPU锁防止并发实验。入口检查分支、df和相关du；默认至少保留10GiB空闲，新目录默认1GiB保护上限。只存向量、JSON和小日志，不保存帧图或复制视频/权重。空间不足停止，绝不清理。HF完全离线，权重失效明确报错，不随机初始化回退，不读deepseek.env或SSH密钥。

## 7. 统计、验收和交付

只有阶段全部答案完整才离线读取旧标签，保留非法答案分母。比较uniform32与air_cap32：准确率、救回/误伤/净增、视频聚类CI（单视频smoke不作泛化结论）、实际选帧数、视觉token、选择/分析/回答/预处理耗时、显存、fallback和解析失败。完整费用包含CLIP与评分调用，不能把最终回答token不增写成总成本不增。

必须通过：候选优先级不丢失、累计预算停止、迭代上界、去重与边界、无正分回退、严格评分解析、实际时间轴、原生等价、缓存身份漂移拒绝、普通断点恢复、未完成不读取GT、无运行标志禁止GPU、旧模块无引用。

## 8. 已完成工程与CPU验收（2026-11-03）

- `air_selection.py`：独立纯CPU选帧，不读标签；固定随机种子的GMM、事件/区间排序、局部补采样和累计预算停止。
- `air_mage_adapter.py`：保留原生两级缩放、真实帧号/时间、完整四选项；异常后恢复处理器，不修改原模型文件。
- `air_protocol.py`：来源/源码/模型/配置身份、只写一次文件、封存记录、逐答案账本和空间保护。
- `air_runtime.py`：本地离线Mage/CLIP；逐帧评分立即落盘、共享视频特征与问题向量、逐轮轨迹、原生均匀输入回归、嵌套tqdm。普通中断重放已保存评分，不重新生成；未完成的视频粗CLIP扫描可能重做该视频扫描。
- `air_analyze.py`：完整后才读标签；非法答案保留分母；救回/误伤、分时长、聚类CI、完整选择成本。各阶段另存`units_snapshot.json`及SHA归档，后续扩跑追加全局账本不会改变旧阶段汇总。
- `air_run.py`、`run_air_frames.sh`：仅开放smoke3/dev30，复用既有GPU锁，用户显式启动；所有答案完成后run直接跳过模型加载。

最终CPU检查共**67项通过**：64项选择/适配/断点/统计控制测试在已有VLMEval科学环境运行；3项真实Mage AutoProcessor测试在Mage环境运行。后者只用临时RGB单元测试夹具，不解码真实视频、不加载模型权重，不构成新增数据集。包含均匀32帧/非均匀3帧的全部最终tensor等价、单图评分入口、原帧号/秒数、处理器恢复、逐轮中断回放、缓存漂移拒绝、未完成禁止读取真值、扩阶段后重复汇总不变。Python语法和shell语法检查通过。

检查证据保存在本轮`engineering_checks/`，包含测试日志、真实CLIP tokenizer的180题完整性报告和最终源码哈希。早期失败检查日志也保留，最终通过结果以`final_verification_v1.json`所列日志/哈希为准。

Mage环境已有NumPy/SciPy，缺sklearn/joblib/threadpoolctl；已用现有科学环境验证对应版本。`prepare`仅在缺失时安装scikit-learn1.9.1、joblib1.6.0、threadpoolctl3.7.0，使用`--no-deps --no-cache-dir`，不升级Mage栈。助手没有执行安装或下载。已有CLIP权重约1.6GiB；当前可用约33.5GiB，新AIR检查产物不足1MiB，旧结果保留原位。

**仍待用户验收：真实视频解码、真实GPU生成及Mage能否稳定输出评分。CPU测试不代表真实实验已通过，更不证明涨点。** 最初smoke为同视频3题，只验接口/缓存/轨迹，不以3题准确率选择规则。若有输入不等价、非法评分或身份漂移，停在诊断，不绕过检查、不自动扩样。

## 9. 用户首轮执行命令

在AutoDL终端执行；首次prepare会联网获取缺少的3个CPU包，不下载模型。支持普通中断后原命令重跑，评分/答案恢复到最后一条完整落盘记录。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VQA/mage_vl/air_v1/run_air_frames.sh prepare smoke3 &&
bash VQA/mage_vl/air_v1/run_air_frames.sh test-cpu smoke3 &&
bash VQA/mage_vl/air_v1/run_air_frames.sh init smoke3 &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VQA/mage_vl/air_v1/run_air_frames.sh run smoke3 &&
bash VQA/mage_vl/air_v1/run_air_frames.sh summarize smoke3
```

首轮汇总：`VQA/eval_results/practical_mage_v1/18_air_frames_v1/smoke3/summary.json`。先核查审计、评分解析、实际帧数、fallback和选择总成本，再决定是否扩现有dev30的90题。不自动进入dev180、val或全量，不触碰字幕/Codec旧结果。

## 10. native_uniform_fix1修订与故障归档（2026-11-03）

用户首轮在0/6答案处被原生索引验收拦截：视频Bkheu99K5lY共13356帧，fps=29.97002997002997。原因是本工程均匀helper使用NumPy整数linspace向下取整，而Mage原生`select_frame_indices`使用`torch.linspace(...).round().long()`。不是权重/CUDA故障，也不是A.I.R.效果结论。首版测试只校验内部同一helper和适配tensor，漏掉了helper与原生采样函数的直接对照；该验收缺口已修正。

- helper延迟导入Torch，明确仅用CPU，逐项复用原生round采样语义；视频短于预算时保留全部帧。基线校验和无证据回退共用该规则，不修改原生Mage。
- 新增真实原生函数与helper的60组帧数/预算对照，包含13356、半整数和短视频；完整原生处理器tensor测试也改用失败视频的帧数及fps。实际Mage环境完整**68项CPU测试通过**，未解码评测视频或加载模型权重。
- 保留原`candidate_v1.json`、旧签名缓存、runtime.json、run_manifest及失败日志，不删除或覆盖。新候选`candidate_native_uniform_fix1.json`，新缓存签名，新阶段目录`18_air_frames_v1/native_uniform_fix1/{smoke3,dev30}/`。
- 依赖准备补齐pip探测/ensurepip，并检查sklearn所需narwhals及joblib所需cloudpickle。锁定5个CPU依赖版本，保留Torch/Transformers/NumPy/SciPy；runtime记录新增两库版本。用户已自行补装pip和narwhals，助手未执行安装。
- 索引不一致仍硬失败，新增count/fps/expected/actual诊断信息，不绕过审计。其他选择配置、提示词和预算保持不变。

当前证据：`engineering_checks/native_uniform_fix1_verification.json`及其指向的68项测试日志。首轮命令沿用第9节，输出路径以本节为准；仍由用户启动真实GPU smoke，不自动扩跑。文档字符串、语法检查和原生源审查不能替代真实视频输入验收。

## 11. smoke3真实运行复核（2026-11-03）

6/6答案完成，输入审计通过，非法答案/评分均0；均匀32和AIR均3/3，净增0。2题没有正分而原样回退，只有1题真正选帧后重新回答，因此仅证明链路可运行，不证明涨点。共168次逐帧分析，推理265.84秒；CLIP共享特征27.24秒；整次运行约5分39秒，不能将回退后的持平计为选帧成功。

日志有20次H.264 mmco短参考帧管理报错，未中止解码，无fallback解码异常，原生均匀tensor一致；这些检查不能证明所有扩展选帧均无像素损坏。保留视频和日志，不批量重编码。可按冻结配置扩已有dev30的90题（复用smoke3），重点记录fallback、分数分布、救回/误伤和全部成本；不修改阈值、不进全量。

## 12. dev30离线误伤诊断（2026-11-04）

90题两路均56/90，救回7/误伤7；两个非法评分属于375题，与14道改变量题不重叠，旧审计保留未通过。已离线审阅全部14题所选帧和评分，并保存小型对照图与来源哈希。核心风险为单帧相关性冒充充分证据、局部扩展强化错误条件、正分帧整体替代均匀输入造成全局上下文丢失；418/484另外暴露已拿到画面但计数/排列仍错的限制。

详见[完整逐题证据及最小方案](ANALYSIS_air_dev30_rescue_harm_20261104.md)。拟只验证一个32帧候选：16个固定全局锚点＋最多16个去重局部代表，不足补均匀帧；复用现有评分，仅重新生成实际输入改变的最终答案。当前仅提出并冻结方向，尚未实现/推理，不承诺涨点，不在14题上手工设规则。先不扩180或全量。

## 13. 冻结候选扩样终评

新增90的hybrid53低于uniform54，未延续总体收益；合计180仅111对110净+1。两阶段审计通过；中等60→65/96，长50→46/84，暂不扩全量。已保存8题CPU画面对照及来源哈希，详见PLAN_air_hybrid_extension90_v1.md的完成复核。原代码/预测冻结保留，未启动新推理。

## 14. 证据到位诊断

基线继续冻结保留。已完成独立12题人工式画面定位和工程，9明确/2有歧义/1视觉不足分开统计；15CPU测试及12题真实处理器验收通过。未启动新QA，不把助手审阅包装为独立人类认证上界。见PLAN_mage_oracle_evidence_diag_v1.md；用户运行run_oracle_diag.sh，待结果再决定第二步。

## 15. 证据到位诊断结论

明确证据9题从uniform3/hybrid4提升到6，三路共同错4题修复3；支持继续做小规模自动片段证据验证可行性，不代表方法已涨点。计数/雕塑/采访位置仍失败，不能假设正确选帧足以解决所有错误。详见PLAN_mage_oracle_evidence_diag_v1.md完成判断。基线/原结果冻结保留，第二步尚未实施。
