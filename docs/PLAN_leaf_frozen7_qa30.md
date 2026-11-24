# 冻结7题新画面的QA诊断，完整30题报告（2026-11-05）

状态：用户已完成真实输入审计和7题QA，完整30题与旧noobs持平、无新增收益；实际结果见文末。以下保留设计和准备记录。

## 1. 固定问题与比较对象

延续condition_leaf_localization_probe_v1已冻结的完整开发30题：

- 7候选：325、450、424、669、389、625、484。使用已保存的result.indices；不看本轮答案再删掉484等失败案例。
- 4锚点：265、549、454、406，保持原noobs输入与答案。
- 19回退：同原范围，保持原noobs输入与答案。

本轮名称为frozen_leaf7_fallback23。这是人工已见开发题范围内的诊断，不是通用自动路由。原先的绘画误检、未覆盖完整硬币变化、4帧采样不含精确峰等问题均保留，不同时改进以混淆比较。

冻结的7题输入中，424为29帧、389为31帧，其余32帧；仍是最多32帧预算，不补齐重复帧。保持原问题+ABCD、ANSWER_HEADER、Mage原生处理器、真实原帧时间、每帧max_pixels150000/min_pixels3136、贪心max_new_tokens32、seed0和原模型身份。

报告三路：原生uniform32、既有noobs、frozen_leaf7_fallback23。主要比较新方案相对noobs的救回/误伤/净增；相对uniform的总差异需同时给出历史noobs相对uniform的差异，不能把复用的已有收益归因新模块。

## 2. 来源冻结与原成果保护

新结果统一放在：

`VLM/eval_results/practical_mage_v1/18_air_frames_v1/leaf_frozen7_qa30_v1/`

新建独立.py/.sh与CPU测试。原定位、解析、QA脚本和全部历史缓存/汇总不修改。候选协议锁定：7/4/19范围、完整30题、原QA文本、逐题冻结帧、定位结果/曲线/文本向量来源SHA、旧noobs和uniform逐题记录SHA、本地Mage身份、处理器/适配器/统计源码与依赖身份。

仅读取旧noobs已审计答案用于明确授权的复用和评测，不向Mage提供这些答案、旧观察、查询编译内容或人工审阅文字。定位结果仅提供源帧索引，不把短观察、事件状态或关系描述加入QA文本。

## 3. 运行分两步，先验证再推理

**audit-inputs（30题，无模型生成）**：加载本地AutoProcessor，不加载模型权重；对每题重建原生uniform32与旧noobs选帧输入。复用既有input_pair/selected_video_inputs路径，检查原生和适配器完整Tensor一致、原问题/选项、原视频和fps、原帧patch位置、grid、视觉token与输入token。新重建的完整uniform/noobs张量SHA必须逐题与旧noobs记录相等，不能仅凭帧号相同复用。

每题审计后原子保存小型audits记录，不保存像素张量或视频切片。中断后跳过已审计题。30题审计齐全且一致后，才允许run；失败保留原记录，不自动重跑/替换旧答案或放宽哈希。

**run（最多7次新生成）**：先将23题按审计记录精确复用，逐字保留原noobs raw_text/pred和输入身份，新增生成计数为0。7候选只使用已冻结新帧，重新验证原生输入与来源审计一致，使用原QA文本生成。单题原生成输出先原子落盘，再形成逐题记录；若中断发生在生成已保存而记录未保存之间，续跑直接补记录，不重复生成。

已保存生成不会因答错/非法答案重试；非法输出保留并在完整汇总中计为错误，同时报审计失败。中断发生在生成输出保存之前的未完成题可重做，日志保留；不宣称GPU物理尝试数在任意崩溃下都恰好7次。已完整时跳过模型加载。GPU共享锁防止与其他实验同时占卡。

## 4. 汇总和结论边界

只有30题审计及30条预测完整、来源校验通过后，summarize才读取独立evaluation_labels.json，核对dataset_lock中的SHA；init/preflight/audit/run禁止读取真值。

- 完整30题：三路正确数/准确率，配对救回、误伤、净增、delta_pp。
- prior12与extra18分别报告；单列中等/长视频及candidate7/anchor4/fallback19。
- 使用既有按视频聚类bootstrap 95%区间，McNemar仅作逐题辅助统计。小样本、已见开发题，不宣称独立泛化或统计显著。
- 23题与noobs相比必须无任何答案变化；若发生变化即拒绝汇总。
- 保存predictions.json、summary.json、逐题来源/输入审计、7条生成缓存、日志和完整记录。包含484失败案例，不按结果改路由。

新增输入审计CPU耗时、候选预处理、7次生成、逻辑token、峰值显存与历史定位/编码/解析成本分开。输入身份审计有额外解码开销，不是部署速度测试；缓存复用不是历史处理免费。历史成本报告保留来源，不重复累加不相关的旧实验成本。

若无净收益，结束本轮并保留负结果，不继续在30题上反复挑规则。若有正收益，仍需冻结后在新增题验证，不能直接全量或将人工7题路由包装成通用方法。

## 5. 用户终端入口（待代码与CPU验证完成后使用）

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_leaf_qa30.sh test-cpu &&
bash VLM/mage_vl/air_v1/run_leaf_qa30.sh init &&
bash VLM/mage_vl/air_v1/run_leaf_qa30.sh preflight &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_leaf_qa30.sh audit-inputs &&
bash VLM/mage_vl/air_v1/run_leaf_qa30.sh run &&
bash VLM/mage_vl/air_v1/run_leaf_qa30.sh summarize
```

入口均先查分支/df/du，剩余空间少于10GiB或AIR结果目录超过1GiB即停，不自动清理。所有新缓存/临时文件在独立结果目录。离线本地模型，无安装/下载/API或env读取。CPU审计与QA均带tqdm，并按题恢复。

前序：[定位画面审阅](ANALYSIS_leaf_localization_probe_20261105.md)、[定位诊断计划](PLAN_condition_leaf_localization_probe.md)。


## 工程准备与实际验证已完成

43项CPU回归通过，其中4项为真实Mage原生处理器的合成RGB完整Tensor回归，不加载权重或解码真实视频。实际来源预检确认7题新输入、23题旧答复用，484保留；424旧30帧→冻结新29帧，389旧32帧→冻结新31帧，其余候选新32帧，不填帧。

模拟完整30题流程验证了30次源审计、7次模拟生成、23次精确复用、生成已保存/记录未保存处中断恢复、再次运行不加载模型、完整30题汇总和prior12/extra18拆分；模拟结果不会保存在真实audits/generations/records目录。输入审计或QA发生异常会按题保存failures诊断，不接受错误审计、不自动换旧答，汇总保留失败事件来源。

原生成文本/选择/旧结果哈希保持不变。真实视频输入审计与Mage生成仍未执行；验证在新结果目录engineering_checks/final_verification.json。新缓存/临时目录均在leaf_frozen7_qa30_v1中，剩余空间约33GiB。下一步由用户按上述命令执行，先看到“源输入身份审计（不生成）”30题，再最多7题新QA。


## 实际QA结果：持平，无本轮新增收益

用户完成30题完整源输入审计、7条新生成和23题精确复用，非法答案0、失败事件0。uniform32为16/30（53.33%），旧noobs18/30（60%），本轮仍18/30（60%）。相对旧noobs救回0、误伤0、净增0。7题新输入Tensor SHA均不同于旧noobs，但7题答案逐一相同，确实重新生成，非误复用。

相对uniform净增2题全部来自旧noobs已有收益，不能归因新增DeepSeek/条件解析/叶事件定位。prior12仍6/12；extra18仍12/18，其中实际改输入3题也均未改答案。候选7题旧/新均6/7；325仍选D、真值C，相关采访画面到位仍未纠正，无法仅据此区分时间理解、人物绑定或其他判断问题。484仍答对不代表此前画面错配已解决。

本次新增7题生成约6.35秒，源输入审计约380.51秒，候选预处理约82.24秒；属于验证开销，不是部署速度基准。历史编码/解析/QA成本单列保留，复用不是历史成本为零。

结论：本配置工程跑通但无新增涨点证据，按原计划归档，不据此扩样或全量，不继续在同30题叠解析、分类、门控规则。低成本noobs与原A.I.R.保留为研究基线；新增模块暂无成为默认配置的依据。

限制：真正改输入仅7题，旧版已对6题。23题固定时，即使7题全对也只到19/30。本轮不能证明所有语义路线无效；bootstrap[0,0]来自观测配对差值全部为零，不是总体等效证明。不能称新增18题全部重新推理验证了新方法。

原数据不改。逐题复核见leaf_frozen7_qa30_v1/results/assistant_outcome_review_v1.json及原summary/predictions。
