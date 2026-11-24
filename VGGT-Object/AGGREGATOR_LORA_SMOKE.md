# Aggregator 后六层 LoRA：增量实现与 GPU 冒烟验证

## 边界

只在原有 VGGT-Object 工程增加可选训练模式；数据集、划分、采样器、深度归一化、损失公式均不改变。F0 / 已完成的 depth_full_pilot_v3 保留。没有下载数据或权重，也没有在本次部署启动 GPU 训练。

## 训练什么

- A16：`aggregator.global_blocks.18..23.attn.qkv/proj`，共 12 个 Linear，rank 16 / alpha 32 / dropout 0.05。A8 为 rank 8 / alpha 16，单独手动选择，不会自动降级。
- 原始 Aggregator 权重（包括所有 frame blocks / Global Attention 基础矩阵）、DINO、Camera Head 均冻结。Point / Track Head 仍不实例化。
- DPT Depth Head 继续全参数训练：32,654,562 参数。
- A16 新增 589,824 参数，合计训练 33,244,386 参数；A8 新增 294,912，合计训练 32,949,474。
- LoRA A 随机初始化，B 全零，初始函数与原模型一致。
- 仅启用 Aggregator LoRA 时取消原来包围 Aggregator 的外层 no_grad。冻结层仍传递输入梯度，LoRA 获得深度损失梯度。F0 保留原有 no_grad 路径。
- 不修改上游 VGGT 源码。当前模型实例对 global 18..23 和随后 frame 19..23 安装非重入 gradient checkpoint，保持随机状态。

## 20 步配置

| 项目 | 默认值 |
|---|---|
| 实验 / phase | A16 / agg_smoke |
| 图像 / 视角 | 392×392 / 4 |
| 微批 | 每次一个物体、多视角 |
| 梯度累积 | 2；有效每步两次物体采样 |
| optimizer steps | 20（不是 20 个微批） |
| DPT / LoRA 峰值学习率 | 2e-6 / 5e-5 |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8，weight decay=0.01 |
| 调度 | 5% warmup，cosine；各参数组最终下限 1e-6 |
| 梯度裁剪 | 1.0，非有限梯度报错 |
| 精度 | Aggregator BF16 autocast，DPT FP32，继承原工程 |
| 损失 | 原深度回归 + 0.05 confidence + 0.1 gradient；MV=0 |
| 数据 | 原 objects_review_v2.json / 原 phase 继承的数据混合；不导入新数据 |

20 步仅验证可训练性，不要求损失单调下降，不证明精度提升。开始、结束会运行原有 validation，因此总时长不只是 20 次反向传播。

## CPU 检查

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
CUDA_VISIBLE_DEVICES='' bash VGGT-Object/run.sh aggregator-check --rank 16
```

在 meta device 检查真实 VGGT 模块结构与本地权重形状；CPU 测试使用官方 Attention Block 的小尺寸实例，验证两步反传、冻结权重逐值不变、LoRA 与 DPT 更新、checkpoint 开关的输出/梯度一致、禁用 LoRA 的输出差异和保存重建。CPU 测试不能替代实际大模型 GPU 验证。

## 有 GPU 后启动（从原始 VGGT 初始化）

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
mkdir -p VGGT-Object/logs
nohup bash VGGT-Object/run.sh train \
  --experiment A16 --phase agg_smoke --run aggregator_lora_r16_smoke_v1 \
  > VGGT-Object/logs/aggregator_lora_r16_smoke_v1.log 2>&1 < /dev/null &
```

```bash
tail -f VGGT-Object/logs/aggregator_lora_r16_smoke_v1.log
```

不自动载入旧 400 步 DPT，以免混淆初始化。若未来明确需要热启动，可显式 `--init-adapter VGGT-Object/experiments/depth_full_pilot_v3/best`，记录在新实验配置；这不属于从原始 VGGT 开始的公平配对实验。已有同名目录拒绝覆盖；中断可原命令追加 `--resume`。

显存不够时手动换新 run：先 `--phase agg_smoke_2v`（392 / 2 views），再 `agg_smoke_336`（336 / 2 views），最后可改 `--experiment A8`。不能把不同设置当作相同预算效果比较。当前 CPU 环境无法承诺显存/耗时。

## 成功与失败证据

新目录：`VGGT-Object/experiments/aggregator_lora_r16_smoke_v1/`。

- `parameter_report.json`：所有可训练名字、参数量、实际 12 个注入位置。
- `train.jsonl`：逐步 loss、两个参数组梯度范数、实际使用的学习率、step 时间、allocated/reserved 显存。
- `gradient_verification.json`：第一步 LoRA 和 DPT 更新；冻结权重没有梯度、不在 optimizer、version counter 不变。CPU 测试另做逐元素严格比较。
- `latest/aggregator_lora.safetensors` 和 `latest/depth_head.safetensors`，配套 model_info / trainer；不保存完整 VGGT 基础权重。仅保留 best / latest，沿用原工程。
- `smoke_verification.json` / `memory_report.json`：固定输入下禁用 LoRA 确实改变输出；重新实例化模型并加载后输出一致（atol=1e-4，rtol=1e-3）；GPU型号、显存、时间及初末 loss。
- `complete.json`：所有验证通过后才写；单有 checkpoint 不能判定全部成功。失败时 memory_report 记录错误/OOM，不能视为成功。

## 后续数据扩大后的预算

400 步仍是小样本基线或试运行预算，不能直接当作覆盖 1,400 个物体的充分训练。每步有效 2 次物体采样时，400 步总计 800 次；若 1,400 个都是训练物体，平均仅 0.57 次/物体。若总数包含 val/test，应改用实际 N_train 计算。

数据源 d 的期望访问次数：`steps × accumulation × batch_scenes × p(d) / N_train(d)`；视角随机采样意味着访问一个物体也不代表看完它全部图片。

例如均匀按物体采样，平均 5–10 次需要约 3,500–7,000 步（仅用于预算估算，最终看 validation）。若人为 Omni 70% / Smithsonian 30%，1,200 个 Omni 平均访问 10 次要约 8,572 步，同时会重复访问小 Smithsonian 集，应防止过拟合。CO3D 接入后重新按三源占比计算。

正式结论需要同一批物体、相同划分/分辨率/视角/有效 batch/优化步数/验证策略/初始化，对比 F0 与 A16。20 步通过后也不自动延长训练。原来 400 步结果保留，不能与未来大数据长预算结果直接相减来证明 LoRA 收益。

## 本次 CPU 验证记录

2026-11-19：34 项 CPU 测试全部通过，真实 VGGT meta 结构与本地预训练权重形状核对通过。rank16 训练参数 33,244,386，冻结参数 1,125,286,930。原 data.py、losses.py、objects_review_v2.json 和旧 depth_full_pilot_v3/best 权重 SHA256 均与修改前一致。完整日志 reports/aggregator_lora_r16_cpu.log。没有执行真实 GPU 的 20 步验证，因此显存、实际耗时、精度收益仍待运行确认。
