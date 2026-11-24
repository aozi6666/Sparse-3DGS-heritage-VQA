# Codec固定13312/2048：val90后自动补齐本地714（2026-11-06）

## 用户目标与固定原则

用户要求在准备好val90后直接进入本地714全覆盖，不再来回调参。主路线沿用dev180冻结的“全部Codec来源时刻＋完整邻句/时间标记、13312字幕token”，对照沿用“最多32个实际来源字幕锚点＋2048”。原生Codec无字幕为配对基线；已有Frames连续字幕2048作为不同输入条件的参考。

不改视觉patch、模型权重、QA提示、预算或选帧策略，不看val/test得分改变路由。实际完整字幕超13312或2048对照无法满足原规则时，停止并保留诊断，不跳题、不暗中截断/扩容。不将714子集与官方2700题排榜。

## 真实检查状态

本地238视频/714题：旧Codec缓存133视频已有；val90的30视频、test225的75视频共105视频缺缓存。现有cv-preinfer和ffmpeg/ffprobe都已安装，不下载数据/模型/软件。新缓存只能由用户执行CPU阶段生成，助手没有启动缓存生成或GPU推理。

仅作参考的全字幕检查发现，105视频中1个test视频的全部SRT合并文本为14234 token。它不是实际Codec包的严格上界（来源选择会改变文本块/标记），实际是否超过13312必须补缓存后检查。不能提前宣称全体预算通过；本入口会在任何GPU推理前检查全238视频，失败则整体停下。

已完成19项合成CPU测试：包括27单元24生成的假模型流程、重复汇总、控制失败阻断、未知历史基线的新样本、开发结果按Tensor复用、缓存只写新目录、残缺缓存不覆盖、预算超限阻断GPU。714输入计划与源码/模型/旧结果来源检查已通过；真实全量输入Tensor审计须等用户CPU补缓存后执行。

## 执行顺序

1. preflight：检查计划、分支与空间；
2. prepare-cache：CPU补105个新Codec缓存，旧133只读；为238视频重建固定字幕证据，全部预算通过后才生成最终protocol/bundle；
3. audit-inputs：CPU核对714题三路原生展开、来源时间、同一视觉Tensor和当前完整Tensor SHA；有历史Codec控制时额外核对历史prefix；
4. seed-dev：180题540单元与22号开发结果逐项Tensor核对后复用，不重新推理；
5. probe：按实际字幕＋视觉输入长度选择最长视频的三题测试显存，答案直接计入全量；若已是复用开发题，使用保留的历史峰值而不是误报0显存；
6. run-val → summarize-val：先生成val90并检查完整性/合法性；不是新独立验证，已诊断历史需披露；
7. val90数据审计通过后，run-full → summarize-full：自动补齐全部714。这里不以val90是否涨点选择是否继续，也不改预算/规则。

最长输入probe可能属于test225，选择只看输入长度、不看标签或问答正确性；全部新实验都是固定覆盖性评测，不宣称新的独立最终测试。没有根据probe答对多少调参的流程。

## 旧结果复用与新基线

全部714共有2142个Codec逻辑答案，其中dev180已完成的540个仅在当前Tensor与旧记录一致后复用。其余最多1602个逻辑答案需要补齐；同题各路输入相同则仅生成一次。9题6144冒烟不直接当作13312全量记录。

dev180有历史Codec控制，核对旧prefix并检查控制答案复现；其余未保存相应历史输入的题生成新的原生Codec配对控制，报告historical_controls_checked，不伪称全体都复现过历史Codec答案。字幕三路仍使用完全相同的当前视觉Tensor。

原dev180结果、13312预算选择依据、旧6144冒烟、Frames字幕成绩和所有负结果原位保留。阶段short219/dev180/val90/test225分别报告，额外给全714覆盖统计；714不能称未见测试或官方完整基准。

## 数据、省盘与计时

新工程：VLM/mage_vl/codec_full714_v1/。
新结果：VLM/eval_results/practical_mage_v1/23_codec_full714_v1/。

plan.json：CPU缓存前冻结计划、来源、各划分、旧参考；
cache/codec/：只包含新生成缓存；cache/build_attempts/和attempt_records/保留失败输出和调用记录；
video_records/：逐视频来源SHA、真实FPS/帧号、实际字幕全文/预算；
preflight/：每次预算检查及被阻断清单；
protocol.json/bundle.json：238视频预算全通过后才生成；
input_audits/：当前三路完整Tensor哈希和视觉布局；
units/：逐题每路原子JSON、原始输出、成本、复用来源；
resource_gate.json：显存探测门禁；
results/val90/、results/full714/：summary、逐题JSON/CSV、分阶段/时长metrics、配对统计、token_cost；
logs/：各阶段日志与pipeline_walltime_*.tsv；
engineering_checks/、source_snapshots/：工程检查与代码快照。

不保存帧预览、视觉Tensor、KV或logits，不复制模型或旧canvas。预计新缓存/数据约0.3—0.8GiB，设置本轮2GiB保护、磁盘至少10GiB保留线；当前约33GiB空闲。不删除旧缓存；工具只接收独立的新输出目录。失败/中断会保留已经生成的缓存和答案，续跑跳过完整记录。若生成完成但原子保存前中断，最多该在途答案重算。

CPU阶段显式CUDA_VISIBLE_DEVICES为空。GPU固定同原4090D、bf16及软件版本；换硬件会停止，不悄悄复用。完整预算不等于总token相等，不能将与Frames差异归于Codec前端本身。

旧Codec预处理记录约3172秒/62视频，平均约51秒/视频，部分181秒。105新缓存粗估1—3小时；CPU全量输入审计约十几分钟，新增GPU约30—90分钟。建议总体预留2—4小时，视频解码/编码可能更慢；此前“全714约1—2小时”只适合缓存较完整的情况。

统计中的inference_s仅新增生成时间，开发复用成本保留historical字段；不能当端到端部署时间。新pipeline额外记录各次阶段wall-time/status，包含续跑；与论文8×B200时间不可直接比速度。

## 终端执行

已发现机器安装screen，建议用它避免SSH断开。下面会自动CPU缓存→预算→val90→补全714，不需中途按得分再次选择。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
test "$(git branch --show-current)" = feat/air-visual-evidence
screen -S codec714 bash VLM/mage_vl/codec_full714_v1/run_pipeline.sh
```

进入后可Ctrl+A再D脱离；screen -r codec714查看。若任务已退出，重复原执行命令可续跑；正在运行时不要重复启动。若预算/资源/输入审计失败，set -e阻止后续GPU，不自动改规则。保存日志再诊断。

也可先在CPU实例单独执行 prepare-cache 与 audit-inputs，然后GPU可用时运行同一个pipeline；完整记录校验后会跳过。不通过预算门禁不能保证本轮714已经可运行，需以用户实际CPU阶段输出为准。


## 实际CPU阶段停止状态

用户已生成105个新Codec缓存，全部238视频已检查；237通过，test225的1wzgMHrkrys完整字幕14217 token超过13312。prepare-cache在预算门禁以非零退出，尚未生成最终protocol/bundle，GPU新问答0、全量新成绩不存在。CPU缓存生成约5738秒（95.6分钟），缓存/237记录均保留，无须重做。原dev180成绩不受影响。后续应先明确预算协议，不能重复原命令期待绕过门禁。诊断见23号目录preflight/budget_1793947999791046336.json。
