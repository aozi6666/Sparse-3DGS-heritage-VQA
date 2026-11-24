# A.I.R.-Mage Frames v1

> 2026-11-05阶段归档：本轮A.I.R.、混合、条件核查与30题QA消融已完成；停止调参，冻结去观察查询候选。30题均匀16、v2为19、去观察18；新增18题去观察与均匀均12对，尚无稳定收益证据。无字幕Frames选帧与字幕模块独立评测，未联合验证。最新结论见[阶段归档](../../../docs/SUMMARY_air_semantic_frames_archive_20261105.md)。以下早期smoke/GPU待运行等描述作为历史实施记录保留。


当前有效修订：`native_uniform_fix1`（2026-11-03）。原生Mage使用`torch.linspace(...).round().long()`，已修复首版误用NumPy向下取整的错误。旧候选/失败日志原位保留；新阶段路径为`18_air_frames_v1/native_uniform_fix1/smoke3/`，新候选为`candidate_native_uniform_fix1.json`，缓存以新协议SHA隔离。当前完整CPU测试为68项，通过于实际Mage环境；真实GPU/视频生成仍待用户执行。

这是独立的无字幕Frames选帧适配，不依赖旧Query-IP，不修改原生Codec和权重目录。先跑通，再看是否涨点；最终最多32帧，不宣称与均匀32帧有完全相同的实际视觉token。

实现及验收计划：[PLAN_air_mage_frames_v1.md](../../../docs/PLAN_air_mage_frames_v1.md)。参考A.I.R.论文和MIT实现，原许可证保留为`LICENSE.AIR`。

本版差异明确登记：本地HF CLIP-L/14替代EVA-CLIP；Mage做分析和回答；CLIP完整问题＋ABCD分为最多75个内容token的块，单位向量均值再归一化，不丢选项；保留VLM正分帧，完全无正分时回退本题原均匀输入；修正原参考实现的排序和边界契约。不是原作者官方结果复现。

## 当前只执行smoke3

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_air_frames.sh prepare smoke3 &&
bash VLM/mage_vl/air_v1/run_air_frames.sh test-cpu smoke3 &&
bash VLM/mage_vl/air_v1/run_air_frames.sh init smoke3 &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_air_frames.sh run smoke3 &&
bash VLM/mage_vl/air_v1/run_air_frames.sh summarize smoke3
```

`prepare`会检查pip；缺失时用Python自带ensurepip补齐。CPU依赖闭包固定为scikit-learn1.9.1、joblib1.6.0、threadpoolctl3.7.0、narwhals2.26.0、cloudpickle3.1.2；仅在导入检查失败时安装，使用`--no-deps --no-cache-dir`，不升级Torch、Transformers、NumPy、SciPy，不下载模型。权重均从已存在的本地绝对路径加载，HF离线。

smoke3是已有dev30池第一个视频的3题，6个逻辑答案；视频、问题都不是新增数据。测试数组仅属于单元测试，不作为评测数据集。每题最多6轮、每轮最多12个新VLM评分帧，评分成本远高于单次QA，不能称为免费选帧。smoke只有一个视频，不发布显著性或泛化结论；已见过的714题不能重新包装为独立测试。入口没有val/test/full选项。

## 结果和恢复

所有新产物位于：

`VLM/evaluation/results/practical_mage_v1/18_air_frames_v1/`

- `candidate_native_uniform_fix1.json`：推理前登记源码、模型、数据、提示词和规则身份；旧candidate_v1保留作失败版本记录。
- `native_uniform_fix1/smoke3/summary.json`、`predictions.json`：阶段全部完成后才读取标签并汇总。
- `logs/`：每次完整运行日志。
- `cache/<协议签名>/scores/<题号>/<原帧号>.json`：每个评分结果立即落盘；中断后不重复生成。
- `clip_features`、`clip_text_features`：轻量特征及时间成本，问题与视频身份锁定；FP16存储后统一恢复归一化，首次／回放相同。
- `selections`、`rounds`：选择结果与逐轮轨迹。
- `native_audits`、`video_metadata`：真实帧号、秒数、原生输入tensor回归。
- `units.jsonl`：逐答案断点，带输入哈希、实际视觉预算和成本。
- `units_snapshot.json`、`archive/`：阶段答案快照、SHA、汇总、逐题结果和完整协议；共用journal后续追加不改变已有阶段归档。

普通中断后原命令重跑会恢复已落盘答案和评分；CLIP粗特征按视频落盘，视频扫描中途打断可能重做该视频扫描。候选输入或源码漂移、记录损坏、残行会停止并保留现场，不截断或删除旧文件。源码或提示词改动需要显式新版本，不能覆盖旧协议。所有答案完成后重跑run跳过模型加载。

每次入口检查df/du，运行中继续检查；默认保留10GiB空闲，新AIR目录1GiB保护上限。只存轻量向量和文本，不保存帧、视频、权重。GPU使用既有practical共享锁，防止旧实验与新实验并行。新HF/Torch/CUDA/Trition缓存都归本轮目录，不读deepseek.env。

## smoke后决策

先看原生输入审计、非法评分、fallback比例和全部选择成本，再看救回／误伤。原生回归失败时立即停；不要绕过检查继续。smoke正常后才考虑已有dev30的90题，不能按3题涨点下结论。dev30入口已准备，但本次不自动扩跑。

首版CPU验收67项为历史记录；当前修正版68项通过，日志/源码身份在`18_air_frames_v1/engineering_checks/native_uniform_fix1_verification.json`。真实Mage处理器测试只用临时单元测试数组，不解码评测视频、不加载模型权重。真实GPU/视频输入和评分格式仍需用户首次smoke验收；没有承诺准确率收益。
