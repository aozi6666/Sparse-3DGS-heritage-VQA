# 自动条件证据核查 v2

目标：先稳定获取短观察和逐条件报告，再检验自动定位的证据是否真正满足题意。不是最终QA，不输出涨点结果。

## 本轮修复

- 不让Mage自由生成复杂JSON。每阶段先生成一句最多64token的画面观察，再输出逐条件两token状态码；程序构造最终JSON。
- U=未知，P=部分证据，E=模型声称充分。G对应全局8帧，0..5对应程序选出的局部片段，J对应联合输入，N表示无引用。合法状态/编号由HF token约束保证，不无限重试。
- 原时间/帧号由程序记录，不解析模型猜出的数字。引用是证据单元级，并非模型已经精确定位某一帧。
- 全部E仍标记语义未验证。限制格式不会修复事实幻觉或充分性误判，不能把“代码可解析”称为理解能力提高。新增重复/截断/空输出检查，原始token与短观察完整保留。

保持v1的8全局帧→6个12秒片段（每片段4帧）→最多32帧联合核查。12秒内4帧仍可能漏掉短动作；首次/最终/顺序的全局完备性仍待审阅。检索使用旧CLIP视频特征、完整问题/选项、显式子事件和新全局短观察，没有人工时间段输入。

## 3题和30题如何安排

固定30题=旧12题＋原开发池哈希选出的18题（9中等/9长视频，新增视频互不重复）。具体题号在candidate.json，未按真值或旧答案选择；仍是开发性诊断，不是独立测试。先运行smoke3：379首次、454事件关系、625多事件顺序，三题包含在30题中，不重复计算。

每题8阶段×2调用=16次，无JSON修复重试。smoke3共48次；扩到30题复用前三题，最多新增432次。单次自然语言观察最多64token；状态码最多2×条件数+1。实际速度待运行测量，不能凭CPU测试承诺时长。

### 先执行smoke3

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_condition_clips_v2.sh init smoke3 &&
bash VLM/mage_vl/air_v1/run_condition_clips_v2.sh preflight smoke3 &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_condition_clips_v2.sh run smoke3 &&
bash VLM/mage_vl/air_v1/run_condition_clips_v2.sh summarize smoke3 &&
bash VLM/mage_vl/air_v1/run_condition_clips_v2.sh check-smoke smoke3
```

读取results/smoke3/review_packet.json，确认短观察是否贴合画面、E是否越过题目条件。如果仍重复或事实不可信，先诊断，不因为状态码合法就扩样。check-smoke只检查工程门槛：每题16调用齐全、观察非空/无明显循环/无占位文字、24段观察中最多6段达到token上限。它不会判断事实真假。

### 门槛通过并审阅后扩到固定30题

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = feat/air-visual-evidence &&
bash VLM/mage_vl/air_v1/run_condition_clips_v2.sh check-smoke smoke3 &&
bash VLM/mage_vl/air_v1/run_condition_clips_v2.sh preflight diag30 &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
bash VLM/mage_vl/air_v1/run_condition_clips_v2.sh run diag30 &&
bash VLM/mage_vl/air_v1/run_condition_clips_v2.sh summarize diag30
```

重复同一命令即可断点继续，逐题/逐片段tqdm及完整日志。每调用、查询向量、窗口、题目断点分别校验来源和代码签名，冲突或损坏停止，保留现场。

## 输出和判断标准

新根目录：`VLM/evaluation/results/practical_mage_v1/18_air_frames_v1/condition_clips_v2/`。

- `candidate.json`：冻结30题与来源、模型、代码签名。
- `calls/`：所有短观察和状态码，原始token/帧/秒/输入tensor哈希和成本。
- `windows/`、`text_features/`、`units.jsonl`：分步断点。
- `results/smoke3/`、`results/diag30/`：阶段汇总与逐题审阅包。
- `engineering_checks/`：CPU测试、数据读取隔离、30题干跑、真实tokenizer/HF约束与9组处理器输入记录。

旧12题仅在自动阶段完成后离线对照人工时间参照；时间命中不是证据充分，更不是答题正确。新增18题没有人工证据标签，审阅包明确标注未审阅，不会编造证据准确率。语义审阅需对照实际视频，重点看首次、最终、顺序、计数是否遗漏必要事件。短文字报告本身不是语义正确的证据。

本轮尚不能比较QA准确率；只有证据报告可信后，才考虑后续固定32帧自动QA。不能将30题格式成功包装成方法成功。

## 省盘和回退

不下载/安装，不复制权重，不保存原始解码帧；复用原图像特征，新查询和输出都很小。所有新缓存/日志在新根目录，沿用10GiB空闲和整个AIR根1GiB保护。记录历史图像编码成本和新推理/文本编码成本；缓存复用不等于部署选择器零成本，原生一致性检查也会产生额外预处理，因此不是端到端速度基准。

旧A.I.R.、混合候选、人工诊断、condition_clips_v1代码和结果均未修改。回退只需停止本入口。

## 完成的CPU验证

29项测试通过（25项新契约测试＋4项原生处理器测试）；真实缓存30题窗口规划通过；3个视频9组处理器输入通过，3组全局输入与原生逐张量一致；真实tokenizer和HF约束处理器4种前缀通过。文件读取拦截对人工定位、GT和旧答案无触发。未加载模型权重或执行生成；真实Mage在GPU上的状态选择和事实质量仍待用户运行。
