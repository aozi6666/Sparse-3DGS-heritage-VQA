# Mage字幕714题收尾消融

固定两路：去编号/时间标记、去相邻补句；三路旧基线只读复用。完整计划：[PLAN_subtitle_ablation_full714.md](../../../docs/PLAN_subtitle_ablation_full714.md)。

结果：`VLM/evaluation/results/practical_mage_v1/20_subtitle_ablation_v1/frozen_r1/`。18项CPU测试、3项原生处理器合成RGB核对通过；真实GPU实验未由助手启动。最多新增1101次生成；327个相同提示须运行时完整输入一致才复用。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
test "$(git branch --show-current)" = feat/air-visual-evidence
bash VLM/mage_vl/subtitle_ablation_v1/run_subtitle_ablation.sh test-cpu
bash VLM/mage_vl/subtitle_ablation_v1/run_subtitle_ablation.sh preflight smoke3
export PRACTICAL_STAGE_CONFIRMED=1
bash VLM/mage_vl/subtitle_ablation_v1/run_subtitle_ablation.sh run smoke3 &&
bash VLM/mage_vl/subtitle_ablation_v1/run_subtitle_ablation.sh summarize smoke3 &&
bash VLM/mage_vl/subtitle_ablation_v1/run_subtitle_ablation.sh run full714 &&
bash VLM/mage_vl/subtitle_ablation_v1/run_subtitle_ablation.sh summarize full714
```

当前CPU机器不能进行真实QA。默认要求历史相同4090D与软件/精度，硬件不同会停止提示，不盲目跨硬件复用。结果是本地覆盖性/事后消融，不是官方2700题成绩；不存在自动根据得分选择新配置。断点按题/路存储；损坏来源/非法答案保留、停止诊断。
