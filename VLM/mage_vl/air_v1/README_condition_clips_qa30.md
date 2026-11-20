# 冻结v2选帧直接QA：30题

冻结condition_clips_v2现有30题的joint_indices，与同题原生均匀32帧比较。不调选帧，不运行CLIP/证据生成/条件判决，不把短观察或条件状态交给最终QA。原问题和四选项、旧QA提示词、模型与像素预算保持一致。

均匀基线全部复用旧来源。每题重新构造原生均匀输入并和适配输入逐tensor比较，检查原视频/模型身份、历史token哈希与grid，再构造冻结新输入。历史未存完整像素哈希，复用依靠来源锁定与确定性原生重建；不把当前tensor哈希当作历史像素校验。任何不符停止，不自动新增基线实验。

最多新增30次最终QA，greedy、seed0、max_new_tokens32，无字幕/API/观察文字输入；不会重跑原480次调用。每题保存原始答案、源基线记录哈希、输入tensor哈希、帧号/grid/token/时间/显存。断点tqdm、GPU锁、空间检查，重复同一命令可恢复；非法答案保留，不自动重试挑答案。

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_condition_clips_qa30.sh init &&
bash VLM/mage_vl/air_v1/run_condition_clips_qa30.sh preflight &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_condition_clips_qa30.sh run &&
bash VLM/mage_vl/air_v1/run_condition_clips_qa30.sh summarize
```

全部新成果：VLM/eval_results/practical_mage_v1/18_air_frames_v1/condition_clips_v2_qa30/。results/summary.json输出两路准确率、救回/误伤/净增、视频簇bootstrap区间及McNemar；按中/长、旧12/新增18分组。results/predictions.json保留逐题真值/预测/四象限/冻结帧号，results/units_snapshot.json归档断点。标签只在30题完整后的summarize读取。

QA新增成本单列，原v2选择/证据生成开销保留在historical_selection_cost，不能将缓存复用写成部署选择器零成本。原生回归检查会增加预处理开销，本轮不是部署速度基准。

这是已见开发子集上的小范围QA检验，不是独立验证或官方完整基准。即使30题正收益也不能直接宣称稳定涨点。旧A.I.R.、混合、v1/v2、字幕等成果保持原位；回退只需停用本入口。不下载、不安装、不复制权重或保存解码帧。
