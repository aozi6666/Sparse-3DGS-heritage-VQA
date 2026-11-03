# 连续字幕2048：test225固定最终评测

本轮锁定现有配置，只对原划分的225题／75个中长视频做一次固定评测，不运行全量，不涉及Agent、API或训练。短视频结果仍来自short219覆盖性评测，不能用本轮冒充短视频独立测试。数据曾有历史基准使用，论文需披露，不能承诺这些视频从未用于其他历史实验。

## 固定输入与三路对照

1. Frames-32无字幕。
2. 原版帧命中、去重、时间排序的对齐字幕。
3. 固定2048文本token预算的连续补句＋时间标记字幕。

沿用冻结配置：32帧、像素上限150000、下限3136、seed=0、贪心解码、最多32个回答token。沿用既有邻句和时间标记函数、字幕提示词，不根据问题答案或视频时长改变规则。字幕预算不是视频＋问题＋回答的总上下文预算。溢出时按既有邻句预算规则处理；若连命中句加时间标记都超限则停止，不临时截断或换预算。

三路重新配对生成，不读取历史测试预测作为基线。每个视频的实际提示词与原生处理路径逐tensor核验，各题记录真实输入token哈希、视觉布局和来源；三路共享同一视频视觉张量。缺失、空白或无命中字幕时精确回退同题无字幕回答，所有225题均进入分母。错误和非法回答不剔除。

## 冻结与统计

候选清单在推理前锁定源码、开发／验证协议、输入划分、字幕源和生成参数。只有全部675个逻辑答案完成后，离线summarize才读取标签。输出总体、中等／长视频、任务类型、字幕状态分组；报告正确数、准确率、救回、误伤、净增、视频聚类bootstrap区间、问题级McNemar，以及视觉／文本／总输入token和新增推理成本。额外原生审计和共享预处理的计时不能直接作为公平部署速度比较。

结果为负也保留，不通过更改预算、按时长选答案或重复挑规则来调整测试成绩。若之后开发新方案，本批test225不再作为新的独立测试集使用。

## 保存、断点与空间

代码入口：`VQA/mage_vl/practical/run_continuous_test225.sh`。

全部成果仍归`VQA/eval_results/practical_mage_v1/`：

- `16_continuous_test225/candidate_v1.json`：冻结清单。
- `16_continuous_test225/cpu_tests_v1.log`、`cpu_checks_v1.json`：CPU检查记录。
- `16_continuous_test225/logs/`：每次运行的完整输出及错误日志。
- `16_continuous_test225/test225_t2048/`：运行清单、逐题预测、最终汇总。
- `cache/continuous_test225/<签名>/`：协议、环境、实际字幕证据、逐答案JSONL断点。
- `07_archive/continuous_test225_fixed2048_v1.json`：汇总后封存报告、预测、协议和来源哈希。

每个答案刷新并同步落盘；普通中断后原命令重跑，跳过已完成单元。来源漂移、损坏或重复单元会停止并保留现场，不自动清理或修复。全部完成后重复run会跳过模型加载。

入口检查df和目录大小，运行中继续检查空间。默认保留至少10GiB空闲，整个practical_mage_v1目录受1GiB保护上限限制。不下载、不复制权重／视频，不缓存帧或tensor，不删除旧文件及缓存。

## 用户执行

CPU初始化已完成，未执行真实推理。675是逻辑答案数，缺字幕回退会减少实际生成次数。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
{ [ "$(git branch --show-current)" = vlm ] || git switch vlm; } &&
test "$(git branch --show-current)" = vlm &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VQA/mage_vl/practical/run_continuous_test225.sh init &&
bash VQA/mage_vl/practical/run_continuous_test225.sh run &&
bash VQA/mage_vl/practical/run_continuous_test225.sh summarize
```
