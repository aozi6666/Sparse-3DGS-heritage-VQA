# Frames-32 + 真实对齐字幕：最小实验入口

此入口不改旧实验代码。只增加本地SRT中与本次Mage原生32帧采样时刻重叠的字幕；不读取全文进行检索、不换序、不加门控，不改模型和生成规则。它是新增字幕模态的配置实验，不能称为纯视觉算法提升。

## 执行范围

- `init`：纯CPU审计/锁定smoke和dev60开发池的SRT来源，无GT、无解码、无推理。已经由助手执行，重复运行内容一致时幂等。
- `preflight smoke`：只读检查，无Torch/模型加载和解码。
- `run smoke`：用户运行，6题。审查新输入与原生处理逐tensor一致，无字幕控制答案应复现；带字幕答案可以改变。必须至少有题实际命中字幕。
- `summarize smoke`：CPU统计，失败即停，不能进入dev30。
- `run dev30` / `summarize dev30`：用户运行90题，与旧Frames32配对；不要只跑原错题。
- `run dev60` / `summarize dev60`：须先查看dev30结果后由用户决定；复用原90题，新增90题另列统计。
- 当前不开放val/test/full，也不自动开发字幕检索。

所有入口核查`vlm`、磁盘与相关目录大小，并激活既有Mage环境。助手未启动GPU实验。

## 先运行这组命令

在远端终端执行；第一步smoke审计通过才进入dev30，完成后停止：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = vlm &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
runner=/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/mage_vl/practical/run_practical_subtitles.sh &&
bash "$runner" run smoke &&
bash "$runner" summarize smoke &&
bash "$runner" run dev30 &&
bash "$runner" summarize dev30
```

任一步失败即停；中断后重跑同一组命令，已完整落盘的题自动跳过，tqdm恢复完成题数。SSH断线不保证进程继续，但已保存题目不需重算。

## 字幕协议

- 从Mage原生video_processor捕获`frame_timestamps`，同时保存真实`patch_positions`原帧号；二者分别保存，不能混淆帧号与秒。
- SRT按UTF-8（可带BOM）读取，支持CRLF、HTML样式、多行和正文内部空行。按照时间轴边界分段，保留原文件；损坏/无法解码时停止，不能伪装成缺字幕。
- 命中规则为`start <= actual frame time < end`，微秒整数比较。同一cue被多帧命中只加入一次，不同时间的相同台词保留；重叠cue全部保留，最后按时间排序。
- 字幕仅插入原Prompt的指令前，问题与选项不变；无有效命中时原Prompt逐字不变。没有扩大时间窗，没有加入未命中的字幕。
- 字幕预算2048个模型token。超过就停止，保留现场，不静默截断、扩预算或换方法。
- 原SRT缺失/为空/没有采样时刻命中时，复用同题旧Frames32原始输出，并明确标记`missing/empty/no_hit`。这是按输入可用性回退，不按GT决定。逻辑成本仍计算该题的视觉回答，不能把复用历史结果当方法加速。
- 无字幕控制只在smoke额外推理；dev每个有字幕题仅一次回答。原Codec无字幕分数仅是历史参照，不能把新增模态收益宣称为优于同模态基线的算法贡献。

## 结果、断点和日志

全部新增数据在：

`/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/eval_results/practical_mage_v1/`

- `09_subtitles/srt_manifest.json`：62个视频（smoke2+dev60 60）的源路径、可用状态、大小、SHA256和字幕段数量。
- `09_subtitles/<smoke|dev30|dev60>/f32_px150000_aligned_sub_v1/`：`run_manifest.json`、`predictions.json`、`summary.json`。
- `09_subtitles/logs/<stage>_<timestamp>_<random>`：每次run独立完整stdout/stderr，包含tqdm、Transformers提示与底层解码器警告。tee配合pipefail保留真实退出状态。
- `cache/subtitles/<signature>/units.jsonl`：每题追加、flush、fsync，题级断点；保存原始回答、Prompt/输入/证据哈希、视觉布局、token、耗时、显存和来源，不保存GT。
- `cache/subtitles/<signature>/alignments/<video_id>.json`：实际32帧索引/秒数、逐帧cue匹配、命中字幕原文/时间与token数。
- `cache/subtitles/<signature>/protocol.json`、`runtime.json`：代码、模型、字幕、数据、原基线、运行环境身份。新动态模型代码缓存继续在既有`cache/hf_modules`中。

统计包含全体、实际用字幕、缺字幕/空字幕/未命中、分时长结果；dev60自动另列新增90题。报告救回/误伤/净差、视频聚类95%区间、非法率与实际新调用/逻辑成本。非法回答保留分母。只有统计脚本读取独立GT文件，推理入口没有GT选择逻辑。

同内容输出幂等；重复key、残缺JSONL尾行、协议/证据漂移会停止，不自动删除/截断。使用原实验根的GPU文件锁，避免本项目实验同时占卡。至少保留10GiB空闲、实验根1GiB软预算，逐题检查；不写抽帧图片/视频副本/完整logits/tensor/KV，不安装新依赖，不复制权重。

## 检查边界

CPU测试和只读预检不能代替GPU smoke。审计检查时序和输入一致性，也不证明所有字幕内容或解码像素正确。如果出现H.264解码警告，保留新完整日志，后续可定位，不删题。
