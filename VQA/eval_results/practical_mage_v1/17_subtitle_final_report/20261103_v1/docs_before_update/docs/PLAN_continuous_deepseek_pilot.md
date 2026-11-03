# 连续字幕最小对照与 DeepSeek 证据复核试验

日期：2026-11-03（Asia/Shanghai）。状态：代码与CPU准备；助手未运行Mage推理、真实帧解码或付费API。目标分支`vlm`，沿用已有Mage环境；不安装依赖、不下载、不删除旧文件。

## 1. 已有结论与本次范围

原Frames-32＋采样时刻命中字幕：dev180为117/180，相对无字幕110/180净增7；val90为52/90，相对无字幕53/90净少1。验证救回7、误伤8，区间跨零，尚无稳定提升证据。诊断记录及原始结果保留，验证负结果另存`07_archive/subtitle_val90_negative_v1.json`。

已核查的具体弱点：第309题两次“Caltech希望去那里上学”均在SRT中、未进入输入；第288、696题也有关键相邻句漏掉、其他选项相关碎句保留。该观察支持改进证据组织，但不能证明修改后必然涨点，也不能证明模型内部具体因哪个词改答。当前val90已参与诊断；后续只在开发池比较候选，test225不开放，不在val90反复挑规则后称独立验证。

这次仅搭建两条分开的实验：

- 最小Mage对照：连续补句＋时间标记，原Frames-32视觉设置保持。
- 外接DeepSeek：独立看真实图片序列与相同连续字幕，再按证据复核；保留API独立回答基线。

不做训练、不恢复门控阈值搜索、不增加Mage帧数、不默认跑全量。两条路线是否有效均须由实际结果决定。

## 2. 连续补句＋时间标记：固定规则

代码：`continuous_common.py`、`continuous_mage.py`、`continuous_analyze.py`，入口`run_practical_next.sh`。

1. 复用旧实验已验证的真实32帧号/秒数和原SRT，原始命中cue全部保留。
2. 对每个命中cue，尝试补前后各一个非空cue；两段之间空白不得超过2秒，避免跨长静默拼接无关台词。
3. 将相邻、空白不超过2秒的cue合并为完整片段，加入`S编号 [HH:MM:SS.mmm --> HH:MM:SS.mmm]`。记录组成cue ID及原始时间，重叠字幕原文保留，不让模型重写证据。
4. 统一2048个现有Mage tokenizer文本token预算，包含时间标记。超限时在整条时间轴上均匀选择完整邻句，记录被省略的邻句；原始命中句不删、不切断。若无可行预算组合则停止，不悄悄扩大预算。
5. 无字幕/空字幕/无原始命中均原样回退原Frames-32输入与回答，不按GT决定回退。
6. 32帧、150000/3136像素、原选项、贪心生成32token不变。新smoke检查无字幕控制答案复现；每个视频首个待处理问题将新Prompt输入与原生处理逐tensor核对。

这是“邻近上下文＋时间格式”的联合对照，不能区分两者各自贡献；增加了字幕访问范围，也不再是标准的仅采样时刻字幕命中协议。与旧对齐字幕、无字幕分别报告，不冒充新的视觉编码算法。

先smoke6，再dev180；dev30入口保留但不是必跑。每题flush/fsync，正常中断后只补缺失题。残行、重复、输入/模型/源码漂移停止，保留现场，不清理。

## 3. DeepSeek：真实帧独立答题与保守覆盖

代码：`deepseek_review.py`、`deepseek_analyze.py`，同一Shell入口的`api-*`动作；密钥占位是`deepseek.env.example`。

### 3.1 能力与输入范围

核查的[官方Vision接口](https://api-docs.deepseek.com/guides/vision/)支持`deepseek-flash`图片输入，使用`https://api.deepseek.com/chat/completions`。当前工程没有原生视频/音频请求。视频以真实图片序列提供，不能写成API看过完整视频。

- 第一轮从原视频解码相同32个原始帧号，时间标记、真实帧ID与JPEG SHA256记录。最长边512、JPEG质量85，图片只留内存，不写抽帧缓存。
- 输入问题、四个选项、连续字幕片段和真实帧。首轮不提供Mage候选、GT、旧错误案例或模型生成描述。
- API返回A/B/C/D或`INSUFFICIENT`、证据充分/完整性标记、逐字字幕引用、实际帧ID、至多两个待查看时间窗，以及简短可观察证据说明。不要求保存私有思维链。
- 仅当独立答案可能改写无字幕Mage时，才申请第二轮复核。计数、顺序、动作题必须先有合法定位时间窗，再从原视频按2fps取局部密集帧，每窗最多8秒、最多两窗/32个密集帧。
- 第二轮仍有全局32帧及同一字幕，增加密集片段帧，并检查独立候选是否满足人物、日期、阶段、计数、顺序、否定等全部问题约束。第二轮不展示Mage候选，只验证独立答案。

### 3.2 固定覆盖规则与边界

以原Frames-32无字幕回答为默认。独立与复核答案须一致，复核明确充分且完整，引用必须存在且范围合法，才允许覆盖。计数/动作/顺序题还必须实际提供密集帧并引用其中的帧；无合法窗口或无密集帧引用就回退。

程序能检查引用原文、帧ID和时间范围，不能证明语义支持或全片计数完整性；局部两窗也不能自动证明全片不存在另一事件。两次使用同一DeepSeek模型，错误可能相关，不称为独立裁判或保证不误伤。系统是否有效主要看真实净纠错与区间。

### 3.3 阶段与公平对比

- smoke：固定pilot清单前3题，只验证接口和保存链路，最多6次正常请求，不据此宣称涨点。
- pilot：开发池中旧无字幕/对齐字幕的所有预测冲突题，加按固定哈希选出的中/长视频各6道预测一致题；选择不读GT。该池偏向冲突，不代表总体Acc。
- dev60：完整dev180，只有人工显式调用才进入，不自动扩展。所有阶段同签名共用已完成单元/响应缓存。
- 报告原Mage、API独立第一轮、Mage＋API复核；保留API拒答分母。组合系统可能增加第二轮和密集帧，其资源条件与独立第一轮不同，效果和调用成本必须一起报告。
- API独立回答不等于“任何情况下直接换API就更好”；若需验证API全量替代方案，应在相同数据上单独固定并测量相应策略。现有API别名可能由服务商更新，原始响应保存返回model信息，不能声称使用永久固定checkpoint。

## 4. 保存、断点和费用控制

所有持久化文件均在`VQA/eval_results/practical_mage_v1/`：

- `11_continuous_subtitles/`：候选、阶段清单、逐题预测、汇总、完整stdout/stderr日志。
- `cache/continuous_subtitles/<signature>/`：每视频新字幕证据、逐题断点、模型协议/运行环境。
- `12_deepseek_review/`：候选、smoke/pilot/dev60运行清单、结果和日志。
- `cache/deepseek_review/<signature>/`：原始API响应JSON、请求SHA256、图片时间/哈希元数据、逐题断点、请求尝试账本。响应中的使用量和返回模型信息保留；请求中的图片base64、Authorization和API key不写日志或缓存。
- `11_continuous_subtitles/cpu_checks_v1.json`：新旧检查结果和源码指纹。

当前API候选为`12_deepseek_review/candidate_v2.json`；v1仅是修正跨阶段汇总幂等性前的未运行准备记录，保留原位，没有实验结果或API响应。两阶段扩大后重新汇总smoke不会因其他题的请求账本增长而改变其摘要。

不复制视频/模型/环境，不保存帧图片、tensor或完整请求base64，不新建Codec缓存。入口复查df/du；默认至少10GiB空闲、结果根1GiB软预算。当前约34GB空闲，新增记录规模小，不执行清理。

API采用标准库HTTP，不安装SDK。只有`api-run`且用户设置`DEEPSEEK_API_CONFIRMED=1`、填写环境变量密钥才联网。接口固定官方HTTPS地址，禁用重定向，不自动重试。API响应先落盘，再做结构/引用解析；解析失败也不重新付费请求。

`--max-calls`是本签名跨阶段累计请求次数上限，包含失败和结果不确定的请求；`--max-total-tokens`是已存响应的用量停止线，可能被最后一次响应越过，不是金额硬上限。超时/断线可能已计费，恢复时默认停止，不自动重复请求；核查后用户可显式加`--retry-uncertain`。明确HTTP错误停止本次运行，修正账户问题后再次运行仍受次数上限约束。

正常中断重跑同一命令，复用已完成答案和响应；前台进程不保证SSH断线后继续。原始响应尚未落盘的极小时间窗不能保证恰好一次计费，上述pending保护用于避免自动重复消费。

## 5. 用户执行命令

CPU初始化与预检已由助手执行，不需要用户先安装任何东西。助手未执行下面推理/API命令。

先跑连续字幕smoke，再直接扩到dev180：

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
git switch vlm &&
test "$(git branch --show-current)" = vlm &&
export PRACTICAL_STAGE_CONFIRMED=1 &&
next_runner=/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/mage_vl/practical/run_practical_next.sh &&
bash "$next_runner" run smoke &&
bash "$next_runner" summarize smoke &&
bash "$next_runner" run dev60 &&
bash "$next_runner" summarize dev60
```

DeepSeek单独从3题smoke开始；终端安全输入密钥，不粘贴到聊天、不记录到Shell命令历史：

```bash
source /root/autodl-tmp/Mage_ViT/mage/bin/activate &&
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA &&
test "$(git branch --show-current)" = vlm &&
read -rsp '输入DeepSeek API Key（不回显）: ' DEEPSEEK_API_KEY &&
printf '\n' &&
export DEEPSEEK_API_KEY DEEPSEEK_API_CONFIRMED=1 &&
next_runner=/root/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/mage_vl/practical/run_practical_next.sh &&
bash "$next_runner" api-run smoke --max-calls 6 --max-total-tokens 400000 &&
bash "$next_runner" api-summarize smoke
```

smoke审计通过后由用户另行决定pilot，不自动执行。扩pilot时需要显式提高累计调用额度，例如`api-run pilot --max-calls 100 --max-total-tokens 4000000`，再`api-summarize pilot`；这些是次数/用量额度，不是费用估算或准确率承诺。完整dev180与test225均不在上述API命令中。

## 6. 完成与未完成

- 已实现：连续补句、预算与时间块、Mage新输入核查、API独立首答、原视频图片输入、局部密集序列、引用范围检查、保守覆盖、独立/API组合统计、断点、请求缓存、次数限制、密钥占位与模拟网络检查。
- CPU检查：33项新增测试及78项旧回归，共111项通过；包括模拟付费请求只发送一次、部分响应恢复、次数/用量停止线、超时保护、密钥不落盘、引用伪造拒绝、密集帧要求，以及跨阶段汇总幂等。真实清单初始化与无网络预检通过。
- 尚未验证：GPU实际输出、DeepSeek实际接口与账户可用性、真实解码兼容性、语义复核能力及准确率收益。CPU测试不能替代这些实测。
- 不在范围：原生视频/音频API、全片密集计数、API金额硬限额、训练、test225、714题全量或保证涨点。

原始参考：[DeepSeek Vision](https://api-docs.deepseek.com/guides/vision/)、[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[JSON Output](https://api-docs.deepseek.com/guides/json_mode/)。现有路线与论文定位见`PLAN_r2v_practical_accuracy_gain.md`、`SUMMARY_heritage_mage_aligned_subtitles.md`。
