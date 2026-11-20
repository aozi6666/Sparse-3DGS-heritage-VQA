# 自动完整条件＋片段联合核查 v1

本入口用于已有 Video-MME 12 道诊断题的小范围可行性验证，不是新的独立测试。旧 A.I.R.、均匀32帧、16＋16混合及人工证据诊断全部保留。本轮只产出自动证据报告，不生成最终选择题答案。

## 固定流程

1. 8 帧全局概览：Mage 观察事实、缺失条件，提出最多4个搜索对象。
2. 复用原全视频1fps CLIP图像特征；编码完整问题及自动搜索对象，多查询轮流检索6个不重叠12秒片段。首次/开头优先覆盖开头两个片段，最终/结尾优先覆盖末尾两个片段。
3. 每片段4帧＋全局8帧联合观察，保留原视频时间与帧号；记录可见事实、满足的条件和缺失证据。
4. 汇集最多32帧及本轮自动报告，再做一次跨片段条件核查。

每题8次逻辑调用，12题共96次。单次最多384个新token；JSON格式失败仅修复一次，最多192次实际调用，原失败输出保留。该预算是有限证据搜索，不能保证找到所有关键事件；12秒内4帧也可能漏掉瞬时动作。

## 使用与恢复

使用 Mage 原虚拟环境和本地权重，无需新增下载。入口会检查分支、空间、缓存身份及代码签名。每次调用、查询向量、候选窗口分别保存并校验；中断后执行同一命令复用已完成部分。损坏或漂移停止，不自动覆盖。tqdm显示逐题及逐片段进度，完整日志另存。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_condition_clips.sh init &&
bash VLM/mage_vl/air_v1/run_condition_clips.sh preflight &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_condition_clips.sh run &&
bash VLM/mage_vl/air_v1/run_condition_clips.sh summarize
```

新成果根目录：`VLM/eval_results/practical_mage_v1/18_air_frames_v1/condition_clips_v1/`。主要文件：`candidate.json`冻结协议；`calls/`逐次原始输出和输入哈希；`windows/`自动片段；`units.jsonl`逐题断点；`logs/`终端日志；`results/summary.json`汇总；`results/review_packet.json`逐题审阅包。缓存和检查记录均在该根目录，不保存原始解码画面或复制权重。保留10GiB空闲及整个AIR根1GiB容量保护，不删除旧文件。

## 隔离与解释边界

自动 init/preflight/run 不读取人工时间段、定位说明、GT或旧答案；真实初始化已用文件读取拦截验证。仅全部12题完成后的 summarize 读取人工定位，计算离线时间覆盖参照，没有回流检索的路径。

JSON合法、引用帧存在、开头/结尾覆盖和跨片段引用只是结构条件。`model_supported_unverified` 是模型声称充分，**不等于语义正确或完整条件已经被证明**；`machine_verified` 始终为false。时间段命中也不等于正确理解。结果先检查是否真正满足首次、最终、顺序等限制，再决定是否值得做后续固定32帧QA。不得将本轮报告当作准确率提升。

部署成本应包含旧CLIP图像特征的计算；汇总同时保存历史共享图像特征成本、新文本编码和所有生成/修复成本。缓存复用不等于部署时没有选择器开销。

## 已完成工程检查（未执行模型推理）

- 27项CPU测试通过（23项新测试＋4项原生处理器测试）。
- 真实旧缓存12题干跑：6个非重叠窗口/题、联合不超过32帧。
- 两个真实视频共6组CPU处理器输入检查通过；全局8帧与原生输入逐张量一致，局部/联合原时间索引正常。
- 人工定位及标签读取拦截无触发；记录在 `engineering_checks/`。

回退时停止使用这个独立入口即可，旧实验入口、缓存与结果未改动。
