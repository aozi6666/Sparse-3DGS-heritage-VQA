# 原生Codec字幕短中长9题冒烟

主候选：全部实际保留patch来源时刻＋6144字幕预算；对照：最多32个字幕锚点＋2048。视觉patch、原生Codec、模型权重保持不变。两字幕条件不是等证据/等总token预算。

完整计划：[PLAN_codec_subtitles_smoke9.md](../../../docs/PLAN_codec_subtitles_smoke9.md)。结果：`VLM/evaluation/results/practical_mage_v1/21_codec_subtitles_v1/`。

14项CPU测试、9题真实canvas的CPU输入审计已通过；真实GPU问答未启动。只读复用旧缓存，不生成/修改/删除旧canvas；有缺失即停止。最多27逻辑答案，预计24次新生成、3次复用；每题/路原子JSON续跑、tqdm进度。

运行：激活Mage环境后，依次执行 `bash VLM/mage_vl/codec_subtitles_v1/run_codec_subtitles.sh preflight`、`audit-inputs`，用户设置 `PRACTICAL_STAGE_CONFIRMED=1` 后执行 `run` 和 `summarize`。测试入口为 `test-cpu`。本版只开放smoke9，无自动全量扩跑。
