# Cultural LoRA（Mage-VL 域适配）— 归档说明

> **归档日期：** 2026-10-15  
> **当前状态：暂停全量微调；先改 codec 选区，再训 LoRA。**  
> 范围：仅 `VQA/`；冻 Mage-ViT，只训 LLM LoRA。不改 3DGS 主线。  
> 代码：[`VQA/mage_vl/cultural_lora/`](../VQA/mage_vl/cultural_lora/)（分支 `Mage_LoRA`）。

---

## 0. 决策归档（必读）

| 决定 | 说明 |
|------|------|
| **不做** | 在「与现网完全相同」的 Mage 默认选区策略上跑全量 Cultural LoRA / 榜测 |
| **原因** | 后续**一定会改**视频 patch→视觉 token 的选区（Question-aware / ego-motion 等），与现 `mage_vl` 默认 codec 少 token **不会一模一样** |
| **顺序** | ① 先落地新选区并评测 Base → ② 再在**同一套新选区前向**下做/重做 Cultural LoRA（训推一致） |
| **冒烟产物** | 仅作管线验证与备份，**不当正式域适配结论**；可保留 adapter，正式实验需按新选区重训 |

一句话：**微调工作排到「新选区就绪」之后；现在只归档管线与冒烟结果。**

---

## 1. 实验结果（优先保留）

### 1.1 基线（已有，非 LoRA）

| 设置 | 指标 | 路径 |
|------|------|------|
| Mage-VL Base × Video-MME（codec+HEVC，32frame） | overall **63.7%** | [`VQA/eval_results/Video-MME.md`](../VQA/eval_results/Video-MME.md) |

### 1.2 Cultural LoRA 冒烟（已完成，2026-10-15）— 管线验证

| 项 | 结果 |
|----|------|
| 数据 | MSVD **8k** + CiQi **16k**（seed=42，硬抽硬删后） |
| 步数 | **1000** step；约 **11.5 min**；`oom_skips=0` |
| 超参 | r=16, α=32, dropout=0.05, lr=1e-4, grad_accum=8, bf16, max_frames=8, max_pixels=150k, max_seq_len=3072, gradient checkpointing |
| 产出 | adapter **≈132 MB**（仅 PEFT） |
| 校验 | `verify_disk_policy --stage after_smoke` **通过** |
| 路径 | `VQA/mage_vl/outputs/cultural_lora/smoke/adapter/` |
| 复检 | `smoke_fitcheck`（200 step）：减帧/降像素保答案，无截光跳过 |

> 归档备注：冒烟在**默认** Mage 选区/帧采样设定下完成；**正式 Cultural LoRA 须等新选区后再训**，勿把本 adapter 当最终域适配权重写进论文主表。

### 1.3 尚未产出（等新选区后）

| 实验 | 状态 |
|------|------|
| Cultural LoRA 全量 1 epoch（新选区前向） | **延期** |
| OmniVDCBench `heritage_zh`：Base vs LoRA | **延期** |
| Video-MME overall：LoRA 复测 | **延期**（Base 63.7% 已有） |

---

## 2. 实现技术（已落地，可复用）

| 项 | 选择 |
|----|------|
| 基座 | 本地 `VQA/mage_vl/models`（`local_files_only`） |
| 框架 | PyTorch + PEFT；`train_lora.py` 自建 loop |
| 存盘 | 只存 `adapter/` |
| 显存 | gradient checkpointing；超长则减帧/降 `max_pixels`，**不裁答案** |
| 环境 | `VLMEvalKit/vlmeval_env` + `uv pip install --python ... peft duckdb` |

恢复工作时：新选区钩子接上后，训练前向走同一套选区，再跑全量命令（见 §5）。

---

## 3. 微调策略与固定权重

| 项 | 设定 |
|----|------|
| **冻结** | `model.visual`（Mage-ViT）及非 LoRA 参数；选区逻辑本身默认不训 |
| **可训** | LLM LoRA：`q/k/v/o_proj` + `gate/up/down_proj` |
| LoRA | r=16，alpha=32，dropout=0.05 |
| 优化 | AdamW lr=1e-4；batch=1；grad_accum=8 |
| 冒烟可训量 | ≈ **33.0M / 4.77B（0.69%）** |

关系澄清：LoRA 挂在**已训好的 Mage-VL**上，不是裸 Qwen；改选区一般**不必**重训 ViT，但正式 LoRA 应在**新选区**下训推一致。

---

## 4. 微调数据集（已准备，保留）

| 来源 | 用量 | 路径 |
|------|------|------|
| MSVD-QA | **8k** QA | `VQA/mage_vl/data/MSVD-QA/train_8k/` |
| CiQi-VQA | **16k** | `VQA/mage_vl/data/CiQi-VQA/`（`train_16k.jsonl` + 子集图） |

合计 24k；seed=42；原全量已硬删。评测集仍为 OmniVDCBench `heritage_zh` + Video-MME。

---

## 5. 阶段（更新后）

```text
[已完成] 子集硬抽硬删 + verify
[已完成] 冒烟 1k + fitcheck 200（管线归档）
[进行中/下一步] 改 codec 选区（≠ 现网默认）并在 Base 上评测
[延期]     新选区就绪后 → Cultural LoRA 全量 1 epoch
[延期]     heritage_zh / Video-MME：Base vs LoRA
```

全量（**延期后**用，且须已接新选区）：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/python VQA/mage_vl/cultural_lora/train_lora.py \
  --model VQA/mage_vl/models \
  --max-steps 24000 \
  --output-dir VQA/mage_vl/outputs/cultural_lora/full_1ep
```

---

## 6. 磁盘与 4090 时间

| 项 | 约占用 / 时间 |
|----|----------------|
| 基座 + MSVD8k + CiQi16k | **≈18.6 G** |
| 冒烟 adapter | **≈0.13 G** |
| 冒烟 1k | **≈11.5 min** |
| 全量 1 epoch（延期） | 估 **5–12 h** / 卡 |

---

## 7. 相关路径

| 路径 | 说明 |
|------|------|
| `VQA/mage_vl/cultural_lora/` | 准备 / 训练 / 校验（已就绪，暂停全量） |
| `VQA/mage_vl/outputs/cultural_lora/smoke/adapter/` | 冒烟归档（非正式结论） |
| `VQA/mage_vl/models/` | 基座只读 |
| `VQA/eval_results/Video-MME.md` | Base 63.7% |
