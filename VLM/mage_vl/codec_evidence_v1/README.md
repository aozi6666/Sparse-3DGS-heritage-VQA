# Codec＋固定字幕＋4张补充原帧：先看效果

日期：2026-11-09。分支必须是`feat/air-visual-evidence`；所有入口都会检查，发现不符直接停止，不自动切分支。

代码已放在服务器：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VLM/mage_vl/codec_evidence_v1`。

结果目录：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VLM/eval_results/practical_mage_v1/codec_evidence_v1`。

## 已完成与未完成

- 已冻结开发180题/60视频和Codec候选池；固定种子20261109，中长视频各取10个，共20视频/60题作为pilot60。其余120题不会自动启动。
- 已完成6种联合输入CPU审计：纯Codec、1图、2图、图片交换、图片在前、4图；逐槽文本/像素/网格/位置一致，纯Codec完整Tensor复现旧A。
- 最终代码通过14项无GPU回归测试；极小随机BLIP配置通过CPU API检查。未加载BLIP预训练权重、未运行新QA。
- 未验证真实联合生成；没有新增准确率、性能或涨点结论。GPU探测必须先运行。
- 初次CPU校验器漏取首个Codec时间标签的失败与构建清单保留在`build_archive`，原生处理器未修改。

## 省盘与保留策略

- 原视频、字幕、旧预测、Codec canvas与来源数组全部只读，不复制、不删除、不重编码。
- 不保存批量RGB/JPEG、不生成新视频；解码只在用户启动GPU阶段时进行，画面留在内存。
- 新增保存候选索引/覆盖率、每批分数与RGB哈希、逐题答案和审计。结果目录上限256MiB；磁盘空闲不足10GiB即停，不清理文件。
- BLIP单独下载一个PyTorch权重，约1.78GB十进制，忽略TensorFlow权重；不同时下载多种底模。模型目录不计入256MiB结果限制。
- 文件原子写入，已有冲突或损坏断点会停止并保留现场，不自动覆盖。下载、打分批次、GPU探测和问答都可重跑续接。
- `tqdm`显示批次/答案进度。`run_pilot.sh`自动保存每次日志。
- 不安装或升级现有环境。CPU准备单线程、低优先级，GPU隐藏；GPU阶段仅由用户启动，发现已有GPU计算进程立即退出，不等待、不杀进程、不抢占任务。

## 终端操作

在AutoDL终端执行。已有环境已通过检查，当前不需要pip安装包。

### 1. 下载固定BLIP模型（不使用GPU，可断点重跑）

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 CODEC_EVIDENCE_DOWNLOAD_CONFIRMED=1 bash VLM/mage_vl/codec_evidence_v1/run.sh download-blip
```

下载到`/root/autodl-tmp/models/blip-itm-large-coco`。首次冻结仓库revision，再只下载必要文件；下载完成后记录文件SHA-256。重复执行同一命令续传。若镜像不可用，保留目录，将同一命令的`HF_ENDPOINT`改为`https://huggingface.co`重试，不改revision、不重复建另一份权重。

镜像用法依据[HF-Mirror](https://hf-mirror.com/)；按文件筛选和revision固定依据[Hugging Face下载文档](https://huggingface.co/docs/huggingface_hub/en/guides/download)。本次未从AutoDL验证镜像下载速度。

如果后续确实提示缺包，先保留完整错误，按缺失包单独处理；不运行`pip install -U transformers torch`。需安装普通缺失包时可使用清华源`https://pypi.tuna.tsinghua.edu.cn/simple`，不要把“缺某个包”处理成整环境升级。

### 2. 等现有GPU任务结束后，运行第一轮

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
CODEC_EVIDENCE_GPU_CONFIRMED=1 bash VLM/mage_vl/codec_evidence_v1/run_pilot.sh
```

顺序固定：

1. `probe-gpu`：先复现少量旧A答案，再检查1图/4图真实联合生成，并对开发池旧基线最长输入加图作探测。
2. `score-gpu --scope pilot60`：BLIP对原Codec候选帧打分，batch=8，每视频批次落盘，不生成QA。
3. `run-gpu --scope pilot60`：A精确输入核验后复用，B/C各补4张完整帧；相同完整Tensor可明确复用。
4. `summarize --scope pilot60`：完成后才读取GT，输出准确率、救回/误伤/净增和视频聚类区间。

有任何错误时整条流程停止。重复第2步命令继续，不要同时启动两个本实验。已有GPU计算进程仍在时会明确退出；脚本不会操作它们。

如需断开终端，可由用户先创建screen会话后在会话内执行第2步；脚本本身不会创建或操作已有screen会话。

### 3. 查看状态或重新汇总

```bash
bash /root/autodl-tmp/Sparse-3DGS-heritage-VQA/VLM/mage_vl/codec_evidence_v1/run.sh status
bash /root/autodl-tmp/Sparse-3DGS-heritage-VQA/VLM/mage_vl/codec_evidence_v1/run.sh summarize --scope pilot60
```

状态不加载模型、不使用GPU。范围没跑完整时汇总会提示续跑，不将缺失样本算错或悄悄跳过。

## 冻结的第一轮规则

| 路线 | 输入 |
| --- | --- |
| A | 旧Codec＋旧完整字幕32768上限，逐题原文不变 |
| B | A＋从相同候选池取四个分位中心的完整原帧 |
| C | A＋BLIP相关性贪心选择4张完整原帧，固定2%时长间隔，不足则按分数补齐 |

候选池为原缓存所有`groups[].frame_ids`的并集，按真实帧号排序，不只取实际保留patch的帧。分组交换坏帧导致的帧号乱序不被当作时间顺序。

BLIP查询使用完整原问题和全部四选项；不读字幕、GT或旧答案。文本超限直接停止，不静默截断。完整原帧通过原生图片处理器输入，max_pixels=150000、min_pixels=3136；B/C实际网格和新增视觉token必须一致。

补图仍是局部坐标图片，文本明确其来源秒数；不声称统一视频时间RoPE。原Codec的像素和patch位置必须保持不变。

不加入ROI、对象检测、动态预算、条件解析、答案裁判、Codec重排或D覆盖率重加权。只准备了coverage元数据，D尚未实现/评测。不会自动扩剩余120题或714题。

## 关键结果文件

- `manifest.json`：固定清单、划分、源码/模型身份、候选、字幕和旧A来源。
- `audits/cpu_mixed.json`：6种逐槽CPU对齐验收，零CUDA/零解码/零QA。
- `audits/runtime_cpu.json`：现有依赖和微型随机BLIP API检查。
- `audits/gpu_probe.json`：用户后续运行的GPU探测；当前未生成。
- `score_chunks/<video>/<offset>.json`：batch断点；`scores/<index>.json`：完整逐题分数。
- `units/<index>_<A|B|C>.json`：逐题原始回答、输入身份、实际预算、成本和复用来源。
- `results/pilot60_<hash>.json`：不可覆盖的配对汇总。
- `logs/pilot_<time>_<pid>.log`：每次用户启动的完整流程日志。

历史dev180、val90、test225已经使用过。本次pilot60是新的方法开发试验，不是全新独立测试。是否继续，由C−B、C−A、新增部分的表现和成本共同决定，不能保证涨点。

