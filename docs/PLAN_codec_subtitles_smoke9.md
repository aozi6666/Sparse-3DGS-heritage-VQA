# Codec＋字幕：短中长9题配对冒烟（2026-11-06）

## 决策及预算解释

2048是既有字幕实验人为固定的字幕文本预算，不是Mage上下文上限。模型配置text_config.max_position_embeddings=262144，但配置上限不保证当前硬件能以任意长度推理。增加字幕会增加总输入、耗时及KV等显存，可能补证据，也可能增加噪声；不能保证越多越准。

用户先同意最多32个Codec来源时刻作字幕锚点，又明确要求优先效果。CPU诊断发现长视频所有来源命中字幕正文3107 token，带全部邻句和标记5147 token，2048/4096不能容纳现规则。故本轮保留原2048为小对照，主要候选为全部实际来源时刻＋6144字幕预算。6144仅是有依据的候选，不叫“已知最佳”配置。

上一轮Frames字幕成果不被否定、不重跑、不替换。6144不与Frames2048视为等总预算；若有收益，应归于Codec系统＋额外文本证据这一整体条件，不能宣称Codec视觉前端本身更强。2048/6144还同时改变字幕锚点数量，不是纯预算消融。

## 固定范围

短/中/长各1个已有开发视频，每视频3题，共9题。索引：
0,1,2；219,220,221；483,484,485。

按时长及清单顺序选择，不按新答案或得分选题。每种时长仅1个视频，不能据9题声称涨点、显著性、稳定长视频改进；每时长聚类区间可能退化，不作推断证据。本入口不开放开发集/714自动扩跑。

三路：
- codec_no_sub：原生Codec无字幕，新生成并检查与历史恒等基线答案一致；
- codec_all6144：所有非padding视觉patch实际来源时刻，命中SRT、cue去重，原neighbor1＋时间标记规则，最多6144字幕token；
- codec_anchor32_2048：从上述实际来源时刻中均匀保留最多32个字幕锚点，同规则、原2048预算。

已有Frames-32＋连续字幕2048仅作参考，从历史结果读取，不新增其推理。短视频两种Codec字幕输入相同，运行时完整Tensor核对后复用同题当前答案；预计27个逻辑答案、24次新生成、3次复用。发生非法答案或无字幕控制不复现时保留现场并停止，不自动调预算或扩样。

## CPU预算事实

- 短视频：189个保留来源时刻；两字幕配置均264 token，无省略邻句。
- 中视频：171个时刻；全部6144组实际1533 token，32锚点2048组1519，无省略。
- 长视频：154个时刻；全部6144组实际5147 token，无省略；32锚点2048组2019 token，省略33条候选邻句。

“32锚点”只控制字幕证据采集，不删除/替换/重排任何视觉patch，不是语义选帧。一个时刻有patch被保留不代表整张原帧的内容或完整事件都在输入里。

target_canvas=32是原生配置参数，不保证实际画布数恰好32。三视频旧缓存实际画布36/32/28，保持原生结果，不为凑数裁画布、padding或重排。

## 输入与缓存保护

使用原生HEVC配置 target_canvas32、patch16、group_size32、images_per_group4、mask off、max_pixels150000。视觉编码、patch来源位置、时间展开、模型权重均不改。仅将原生process_codec_video调用透明地约束为读取已冻结旧缓存，后续drop_padding、patch映射、canvas处理、提示时间展开均由原生代码完成；上下文结束恢复原函数。

仅允许三视频已有缓存；缺失、重复、SHA不同或配置不同就停止。不调用缓存生成、重建、删除函数，不对旧缓存建软链接或写新文件。所有新日志/输入审计/预测留在21_codec_subtitles_v1。旧缓存保留原位且只读。

严格要求meta明确fps/帧数，两处元数据一致，不接受原生加载器的缺失FPS默认30回退。patch数组需整数N×3、来源帧号不越界。原始来源索引除以真实FPS对齐SRT的[start,end)，不是用canvas编号或canvas顺序伪造时间。保留原始索引和精确秒数，原生提示仍按原格式时间精度展开。

CPU audit-inputs已对真实旧canvas完成9题审计：无字幕prefix SHA复现，视觉网格/token与历史一致；三路仅文本改变，完整视觉Tensor相同；每路插槽构造与原生处理器完整Tensor一致。此阶段未加载模型权重、未解码新视频、无QA。

GPU run还会重新构建输入与已保存完整Tensor SHA核对。严格要求原torch2.11.0+cu128、transformers5.17.0、RTX4090 D和bf16；换硬件会停止，不暗中跨硬件复用。

## 断点、省盘和数据

结果根目录：VLM/eval_results/practical_mage_v1/21_codec_subtitles_v1/。

- protocol.json / bundle.json：冻结配置、旧来源SHA、三视频字幕原文、全部来源时刻、32锚点、预算省略项、旧参考结果；
- budget_preflight.json：逐视频来源数量和实际文本成本；
- input_audits/：逐题三路完整Tensor SHA、prefix SHA、输入token与视觉布局；
- units/：按题/路原子JSON断点，原始回答、预测、来源、时间、峰值显存与复用证明；
- results/：summary、逐题JSON/CSV、分时长metrics.csv、paired_comparisons.csv、token_cost.json/CSV；
- logs/：每次命令独立日志；cache/仅本轮HF代码、CUDA/Triton及临时缓存；
- engineering_checks/：测试与真实CPU输入审计记录；
- source_snapshots/：代码快照。

复用旧canvas不复制视频/权重/画布，不保存视觉Tensor、KV或logits。设置本轮128MiB保护、磁盘保留至少10GiB，不清理已有缓存。预计新增数MiB。已有完整断点会校验后跳过；在生成完成但原子保存前中断，最多重算该在途答案。只有输入相同才复用，不按置信度或真值挑选答案。

保存耗时是新增生成耗时，旧Codec预处理成本不为零，不冒充部署速度或论文Table5硬件对比。Frames参考成本可从bundle.references及既有17号归档读取，不能把其历史推理记为当前新调用。

## 实施状态

已完成14项CPU测试（纯文本、假缓存、假GPU模型），覆盖锚点子集/半开区间/去重/缺字幕/不接受默认FPS/只读缓存/27单元24调用/续跑/控制失败阻断/重复汇总。真实CPU三路输入审计9/9通过。真实模型权重加载与QA调用均为0，GPU推理交用户执行。

协议签名：10a8e1804926c5c3571915d3805222f7491002893cffdb2bdda017b6523feddc。

建议相同4090D运行预留3—8分钟，已有缓存避免视频重处理；以实际日志为准。只进行smoke9，读完结果后才决定是否扩开发集，不直接跑全量。

## 用户运行命令

```bash
(
set -euo pipefail
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
test "$(git branch --show-current)" = feat/air-visual-evidence
entry=VLM/mage_vl/codec_subtitles_v1/run_codec_subtitles.sh
bash "$entry" test-cpu
bash "$entry" preflight
bash "$entry" audit-inputs
export PRACTICAL_STAGE_CONFIRMED=1
bash "$entry" run
bash "$entry" summarize
)
```

相同命令可续跑。init已由助手执行，仅是CPU准备；不需要另行安装/下载。当前CPU实例不能运行实际QA，必须有GPU。

## 能说明什么

现有Frames字幕成绩仍有效，但仅适用于原协议。新Codec6144若超过同题原生Codec无字幕，可作为额外字幕证据增强Codec系统的初步结果；若超过2048，不能拆分归因锚点覆盖或预算。9题只用于接口/预算/资源/回答合法性可行性，不是方法收益证据。独立增益需要后续固定更大配对实验；不得把两路线涨点相加或用714子集与论文完整2700题排榜。


## 实际冒烟已完成

27单元审计通过，Codec无字幕5/9，两种字幕和Frames参考均6/9。6144 vs无字幕救回2、误伤1；vs2048救回1、误伤1，未证明6144更优。可准备固定dev180输入/预算预检和配对，不直接进入714。见[实际结果](ANALYSIS_codec_subtitles_smoke9_20261106.md)。
