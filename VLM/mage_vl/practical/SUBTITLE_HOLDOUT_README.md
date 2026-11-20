# Frames-32 字幕路线：val90 留出验证

2026-11-03。只验证开发阶段已确定的一个配置；本入口不支持test/full，不自动训练或下载。

## 本阶段做什么

- 固定原视频划分的90题／30视频，与dev60和test按视频不重叠。工程更早有全量研究结果，不能称为从未使用过的数据。
- 主比较仅为当前同配置Mage Frames-32无字幕与真实采样时刻对齐字幕：32帧、150000/3136像素、原选项顺序、2048字幕token上限、贪心生成最多32token。不增加帧数、投票或门控。
- 每题先生成无字幕回答，再生成有字幕回答；缺文件、空字幕、没有命中时，原样复用本次同题的无字幕输入与回答。回退只看输入可用性，不读取GT，不按时长切换答案。
- 两路复用同一组视觉tensor。每个待处理视频的首道题，两个实际Prompt都与原生完整处理逐tensor比较；额外开销是CPU解码/预处理，没有额外控制答案生成。
- 不强行复用旧全量结果或跨协议混分；本次不补Codec推理，也不把旧Codec分数混进val90主比较。之后若主表需Codec，应另补身份一致的基线。
- 字幕模态是标准配置增强，不能把同样配置的原版Mage再次列为不同算法行。

## 存放与省盘

以下路径均在`VQA/eval_results/practical_mage_v1/`下：

- `07_archive/subtitles_dev60_v1.json`：已有字幕dev60汇总与逐题预测快照、开发阶段文件SHA256；原始文件保留。
- `10_subtitle_holdout/candidate_val_v1.json`：在验证推理前锁定唯一候选、源码、原开发来源、输入划分、字幕来源及生成设置。只是验证候选登记，不是看过验证结果后的测试冻结。
- `10_subtitle_holdout/val90/f32_px150000_aligned_sub_v1/`：`run_manifest.json`、`predictions.json`和`summary.json`。
- `10_subtitle_holdout/logs/`：每次运行独立日志，含Python、tqdm和C解码器stderr；失败状态由pipefail传递。
- `cache/subtitle_holdout/<signature>/units.jsonl`：180个逻辑答案单元，单个答案flush/fsync后立即成为恢复点。缺字幕复用也记录一个逻辑单元。
- 同缓存目录`alignments/<video_id>.json`：实际32帧号、秒数、字幕命中与所用文本，每视频一份；另有`protocol.json`和`runtime.json`。
- `10_subtitle_holdout/cpu_checks_v1.json`：CPU验证记录与新旧脚本SHA256。

不复制模型、环境、视频、整部SRT，不存JPEG、视觉tensor、完整logits或KV，不生成Codec缓存；仅缓存当前视频的视觉tensor到内存。动态模型源码缓存继续复用结果根的`cache/hf_modules/`。默认保留10GiB，整个实验根软预算1GiB；每个答案前检查，超限停止且不删除任何文件。新验证持久化主要为小型JSON/JSONL与日志，通常远小于现有34GB空闲；以实际df/du为准。

## 运行

先激活已有环境，再进入项目。下面只运行val90并汇总，助手未执行这段推理命令。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
git switch vlm &&
test "$(git branch --show-current)" = vlm &&
subtitle_val_runner=/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/mage_vl/practical/run_subtitle_holdout.sh &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash "$subtitle_val_runner" run val &&
bash "$subtitle_val_runner" summarize val
```

CPU准备已由助手执行：`bash VQA/mage_vl/practical/run_subtitle_holdout.sh init val`，可重复调用，内容不一致时拒绝覆盖。只读预检为`preflight val`，新单元测试为`test-cpu`。入口每次检查vlm、df/du并激活环境，离线加载现有模型，不安装缺失依赖。

普通中断后重跑同一段运行命令：按已保存答案恢复，tqdm从已有单元计数继续；若一题只完成无字幕回答，仅补字幕回答。完成180/180后再次run直接跳过模型加载。前台运行不保证SSH断线后继续占卡；无需从头重跑。

尾行损坏、重复、源码/模型/环境/字幕漂移、证据错配均保留现场并停止，不能靠删除记录或临时修改脚本解决。运行中的H.264提示可从完整日志定位；原生输入相等和时间命中审计并不证明解码像素或字幕转录完全正确。

## 看结果再决定下一步

只有180个单元完整才读取独立标签文件。非法答案计错且保留分母，审计失败仍保存记录；不自动重跑非法回答。主看`overall.methods.frames32_aligned_sub.vs.frames32_no_sub`中的救回、误伤、净增、提升百分点和视频聚类95%区间，另看`by_duration`、`by_task_type`及字幕缺失分组。

计时分别记新推理调用、逻辑调用与共享预处理；本配对过程包含额外CPU输入审计，不能直接作为部署加速实验。中断前未落盘尝试的耗时也不在单元统计中，应结合完整日志。

val90完成后先如实评估，再决定是否冻结并新增test225入口。即使某时长分组获益，也不根据这些GT临时设计路由后直接在同一验证池宣称独立确认。最终714题含开发数据，不能代替锁定测试结果；本地714题也不是官方完整2700题。
