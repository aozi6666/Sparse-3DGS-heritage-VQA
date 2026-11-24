# Mage证据到位诊断：独立入口

固定12道已有Video-MME题，每组4题（2中等/2长）：选择器误伤、旧方法救回、三路共同错误。不是新增数据集、算法成绩或经独立人类认证的数学上界。

助手逐图定位后，9题标为explicit，626/486为uncertain，704为visual_insufficient。后3题仍保留运行并单独报告，不能把其结果用于宣称“充分证据也答不对”。704另一休息日来自说话内容，未送字幕；626细绳在原预算难分辨；486全片主题仍有叙述依赖。

模型严格保持原问答提示词/选项、32个不同原帧、真实时间、max_pixels150000、greedy、最多32生成token；不送定位说明、答案、字幕、裁剪图或审阅图。旧uniform/AIR/hybrid结果引用原记录。输入文件为inputs_reviewed_v2.json（仅题号+帧号），review_v2.json另存证据评审；v1定位草稿保留。雕塑题在逐图检查后剔除了无关黑白插镜，调整发生在任何新QA之前。

成果：VLM/evaluation/results/practical_mage_v1/18_air_frames_v1/oracle_evidence_diag_v1/
其中review/为离线审阅图，engineering_checks/为CPU证据，units.jsonl为新答案断点，results/summary.json为最终诊断汇总。普通中断重跑原命令；全部完成不加载模型。记录损坏/身份漂移停止保留现场，不自动删除或重跑。共享旧GPU锁，10GiB空闲/整个AIR根1GiB保护。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_oracle_diag.sh init &&
bash VLM/mage_vl/air_v1/run_oracle_diag.sh preflight &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_oracle_diag.sh run &&
bash VLM/mage_vl/air_v1/run_oracle_diag.sh summarize
```

只进行阶段1诊断，阶段2尚未开发。结果优先看9题明确证据和4题三路共同错误是否改善；仅重复答对已有救回题不足以支持新的检索模块。仍错时先检查定位和分辨率是否足够，不能从小样本断言任何选帧方法都无效。

旧A.I.R.、混合候选、扩样入口均不改；回退只需停止本入口、继续使用旧结果。助手不启动QA，真实GPU运行由用户执行。
