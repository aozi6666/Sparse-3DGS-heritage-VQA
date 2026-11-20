# 固定30题：去掉全局短观察查询

本轮只删除v2检索中的短观察查询，保留同30题、全局8帧、6个12秒片段、每片段4帧、最终最多32帧、同一QA提示与参数。只使用原问题＋完整选项＋题目明确列出的子事件。不输入人工时间段、字幕、API、证据报告或条件代码。

直接读取v2中相同原查询的持久化文本向量和旧图像特征，不重新编码，不调用480次证据生成。CPU规划锁定：1题输入不变，29题改变；不变题复用源QA完整输入tensor身份及答案，只对改变输入的29题生成最终答案。所有原基线及v2结果保留。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VQA/mage_vl/air_v1/run_condition_clips_noobs_qa30.sh init &&
bash VQA/mage_vl/air_v1/run_condition_clips_noobs_qa30.sh preflight &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VQA/mage_vl/air_v1/run_condition_clips_noobs_qa30.sh run &&
bash VQA/mage_vl/air_v1/run_condition_clips_noobs_qa30.sh summarize
```

支持逐题断点、tqdm及日志；中断后重复同一命令。新生成沿用已验证的直接QA实现，通过本进程上下文管理器重定向独立目录和校验，不修改旧源文件。新输入每题仍须通过原生均匀/适配逐tensor校验、历史token/grid校验和真实帧号检查。不变输入直接复用既有审计记录；不将缓存复用当作部署零成本。

新根目录：VQA/eval_results/practical_mage_v1/18_air_frames_v1/condition_clips_noobs_qa30/。

candidate.json保存查询、窗口、帧号及来源SHA；units.jsonl为逐题断点；results/summary.json提供均匀/v2/去观察查询三路准确率与相对均匀、相对v2的救回/误伤/净增/视频簇区间；分中/长及旧12/新增18。results/predictions.json保留逐题预测、真值、输入变化标记与查询/帧号；units_snapshot.json归档结果。标签仅阶段完整后读取。

14项CPU测试已通过，全部30题查询及窗口已锁定。新缓存/日志/检查记录在本目录，空间保护10GiB空闲、整个AIR根1GiB。不下载或安装、不复制视频帧/权重。原v2的480次证据开销归档比较，但不是去观察查询部署的必要开销；共享图像和文本编码仍有非零历史成本。本轮仅是开发30题消融，不能据此宣称独立验证、稳定涨点或部署速度。
