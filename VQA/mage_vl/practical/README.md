# Mage 均衡换序实验：当前实现与运行方法

当前实现覆盖计划 P0/P1：来源审计、固定数据划分、四种循环换序、语义投票与概率均值对照、细粒度续跑、日志、汇总、冻结及分阶段入口。尚未运行新配置 GPU 冒烟；不宣称涨点。时间证据验证、VCD 和 CLIP 选帧没有接入。

本实现为独立的 Mage 接口适配，没有复制/执行参考仓库代码，也没有修改 Transformers。相关思路参见 [视频选择偏差研究](https://aclanthology.org/2025.acl-long.162/) 和 [置换一致性与投票研究](https://aclanthology.org/2025.findings-ijcnlp.127/)。本地三个参考工程的路径、commit、审计函数及许可记录在结果根的 `00_sources/audit.json`。

## 默认协议

- 结果根：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/eval_results/practical_mage_v1/`。
- 默认标签：`px150000_f32_frames4_codec1`。每题 Codec 原顺序 1 次、Frames 4 种循环排列，共 5 单元。
- 原文依赖选项位置/标签时回退原顺序，仍计入分母。票数平局优先原答案。
- 保持原 Mage 视频路径、Prompt、像素预算、确定性生成和 `max_new_tokens=32`，避免把只生成 1 token 当成新的“原版基线”。
- 自由生成解析字母用作投票。只在首 token 本身就是解析到的字母时，第一位置 logits 才可参加概率平均；以“答案是...”开头的输出不会被伪造为字母概率。
- 视频预处理使用无标签文本插槽，保持同一视觉输入；smoke 对恒等排列检查全部输入 tensor 与原处理路径逐项相等，并对照旧冒烟答案/grid/时间范围。
- 只缓存每个后端最后一个视频的 CPU 预处理。模型视觉前向仍逐排列执行，不声称实现 KV/视觉编码缓存。
- 跨阶段复用单元。增加 Codec 四排列控制时，使用独立阶段标签但复用相同协议下已经存在的 Frames/Codec 原顺序单元。

## 已准备的数据

- smoke：6 题 / 2 个旧 short 视频，只调通接口。
- dev30：90 题 / 30 个新视频。
- dev60：180 题 / 60 个新视频，包含 dev30。
- val：90 题 / 30 个独立视频。
- test：225 题 / 75 个锁定视频。
- full：714 题 / 238 个视频；最终整体分数包含开发数据，单列独立 test。

`01_manifests/inputs.json` 无答案；`evaluation_labels.json` 仅供离线汇总读取。旧冒烟审计另存无真值摘录。真值不参与换序、投票或每题回退。

## 当前先执行的命令

在 AutoDL 终端：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
bash VQA/mage_vl/practical/run_practical_mage.sh smoke
bash VQA/mage_vl/practical/run_practical_mage.sh summarize smoke
```

初始化已完成，后续重新执行 `init` 会校验相同文件而不覆盖不同内容。纯 CPU 检查可执行：

```bash
bash VQA/mage_vl/practical/run_practical_mage.sh test-cpu
bash VQA/mage_vl/practical/run_practical_mage.sh preflight smoke
```

检查 smoke 汇总的 `audit.smoke_pass=true`，然后再执行：

```bash
PRACTICAL_STAGE_CONFIRMED=1 bash VQA/mage_vl/practical/run_practical_mage.sh dev30
bash VQA/mage_vl/practical/run_practical_mage.sh summarize dev30
```

必要时同样方式执行 `dev60`。不要同时启动 dev30/dev60 或其他 Mage 配置。

## 后续入口（先开发，再选型，不直接运行到底）

```bash
PRACTICAL_STAGE_CONFIRMED=1 bash VQA/mage_vl/practical/run_practical_mage.sh val
bash VQA/mage_vl/practical/run_practical_mage.sh summarize val
# 以下方法名是示例；先根据 val 明确选择，再冻结。不可按 test 结果改选。
bash VQA/mage_vl/practical/run_practical_mage.sh freeze frames_vote
PRACTICAL_STAGE_CONFIRMED=1 bash VQA/mage_vl/practical/run_practical_mage.sh test
bash VQA/mage_vl/practical/run_practical_mage.sh summarize test
# 最后补齐全量，不覆盖已完成单元。
PRACTICAL_STAGE_CONFIRMED=1 bash VQA/mage_vl/practical/run_practical_mage.sh full
bash VQA/mage_vl/practical/run_practical_mage.sh summarize full
```

配置变化必须重新 smoke。`PRACTICAL_CODEC_VOTE=1` 增加 Codec 四排列控制；`PRACTICAL_FRAMES=48` 或 `64` 为独立预算候选。设置后同阶段的运行/汇总/冻结命令都要携带相同变量；默认不启用。

## 断点、日志和磁盘

- 每个完整 `(index, backend, permutation)` 单元写入 `cache/ballots/<signature>/units.jsonl`，立即 flush/fsync。Ctrl+C 后重跑同一命令恢复，当前未落盘单元可重跑。
- 残缺尾行、重复 key、配置漂移和坏缓存均报错保留，不自动修复或删除。
- 阶段目录保存配置/索引、`run_*.log`、汇总与预测；tqdm 显示题数、单元数及当前后端/排列。
- 旧 Codec 缓存只读复用，新缓存仅落在新结果根的 `cache/codec`。不持久化抽帧、MP4、大 tensor、全词表 logits 或 KV。
- 默认剩余磁盘少于 10 GiB 或新目录超过 1 GiB 时暂停供审查；预算是检查点保护，不保证预处理临时峰值精确受限。
- 当前服务器未检测到 tmux，未安装。前台 SSH 中断不保证后台继续，但完整单元可恢复。
- 脚本只阻止本实验的并发 GPU 写入，不能替代对服务器其他任务的人工确认。
