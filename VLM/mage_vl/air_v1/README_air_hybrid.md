# 固定32帧候选 hybrid_g16l16_v1

独立入口，旧run_air_frames.sh及全部结果不变。规则和边界见项目docs/PLAN_air_hybrid_g16l16_v1.md。

16个锚点来自原uniform32的等间隔位置；最多16局部帧来自旧4/5分评分，2秒相邻时间簇轮选、簇内增加时间跨度；不足均匀补齐32。只此一组规则，不按题型、真值或已知错误开关。

已在实际Mage环境通过17项测试（含4项原生处理器CPU测试）。90题规划检查完成：31题复用uniform答案，最多59次新最终QA；不再加载CLIP或生成逐帧分数。每题依然重建并核对输入，记录全部tensor哈希；旧推理未保存像素哈希，所以不把历史来源复用称作历史像素逐位校验。两个非法历史评分明确排除并在报告中保留，不改旧审计。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_air_hybrid.sh preflight &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_air_hybrid.sh run &&
bash VLM/mage_vl/air_v1/run_air_hybrid.sh summarize
```

候选已经CPU init登记；需要重复init也是幂等的。普通中断后重跑上面命令；全部完成则不加载模型。单条记录写后fsync，损坏/残行停止保留现场，不删除或截断。完整日志和逐答案断点在独立目录。

成果根：VLM/eval_results/practical_mage_v1/18_air_frames_v1/hybrid_g16l16_v1/
候选candidate.json；断点units.jsonl；日志logs/；报告dev30/summary.json、predictions.json、units_snapshot.json。原始均匀/原AIR答案不复制权重或特征，只引用源记录和哈希。新HF/CUDA/Torch缓存仍归新成果根；不读deepseek.env。

回退：停止新入口，保留其目录，继续使用旧run_air_frames.sh及native_uniform_fix1/dev30/原结果即可；无须git reset、还原源文件或清除缓存。若以后改规则，必须另开候选版本/目录，不能修改已冻结协议。

注意成本口径：重用历史评分只降低本轮新增计算；报告仍列出历史完整选择成本，不能宣称部署无需评分。90题是已诊断开发集，需报告14诊断题与其余76题，不包装独立测试。此候选收益尚未验证。
