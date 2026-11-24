# Codec固定13312/2048本地714覆盖性评测

先CPU构建缺失缓存并审计预算，再显存probe、val90数据审计，自动补齐714；不按val得分改规则。复用已验证dev180的540答案，不复用不同预算的6144冒烟记录。

计划：[PLAN_codec_full714_fixed13312.md](../../../docs/PLAN_codec_full714_fixed13312.md)。
结果：`VLM/evaluation/results/practical_mage_v1/23_codec_full714_v1/`。
19项合成CPU测试已通过；真实缺失105缓存和全量预算/输入核对待用户执行。固定预算超限即停止，不能静默跳题或截断。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
screen -S codec714 bash VLM/mage_vl/codec_full714_v1/run_pipeline.sh
```

Ctrl+A再D可脱离，`screen -r codec714`查看。执行失败后重复可续跑；正在运行时不要另起同一任务。

单步入口：`bash VLM/mage_vl/codec_full714_v1/run_codec_full714.sh ACTION`；ACTION为init、prepare-cache、preflight、audit-inputs、seed-dev、probe、run-val、summarize-val、run-full、summarize-full。CPU准备不运行模型；CPU补缓存与GPU需要用户设置PRACTICAL_STAGE_CONFIRMED=1，pipeline已设置。测试入口test-cpu只用假工具/假模型。

13312为固定字幕预算，不是模型上限；不知道新缓存实际来源前，不能宣称全部预算已通过。new历史控制缺失的样本会生成新原生Codec配对基线，不伪称旧答案复现。714包含开发/历史诊断，不是新独立测试或官方2700。
