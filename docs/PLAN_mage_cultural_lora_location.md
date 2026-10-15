# Cultural LoRA（Mage-VL 域适配）

> 状态：**冒烟已通过**；全量 1 epoch **待跑**。  
> 范围：仅 `VQA/`；**冻 Mage-ViT**，只训 LLM LoRA。不改 3DGS 主线。  
> 代码：[`VQA/mage_vl/cultural_lora/`](../VQA/mage_vl/cultural_lora/)（分支 `Mage_LoRA`）。

---

## 1. 实验结果（优先保留）

### 1.1 基线（已有，非 LoRA）

| 设置 | 指标 | 路径 |
|------|------|------|
| Mage-VL Base × Video-MME（codec+HEVC，32frame） | overall **63.7%** | [`VQA/eval_results/Video-MME.md`](../VQA/eval_results/Video-MME.md) |

### 1.2 Cultural LoRA 冒烟（已完成，2026-10-15）

| 项 | 结果 |
|----|------|
| 数据 | MSVD **8k** + CiQi **16k**（seed=42，硬抽硬删后） |
| 步数 | **1000** step；约 **11.5 min**；`oom_skips=0` |
| 超参 | r=16, α=32, dropout=0.05, lr=1e-4, grad_accum=8, bf16, max_frames=8, max_pixels=150k, max_seq_len=3072, gradient checkpointing |
| 产出 | adapter **≈132 MB**（仅 PEFT，无合并全量权重） |
| 校验 | `verify_disk_policy --stage after_smoke` **通过** |
| 路径 | `VQA/mage_vl/outputs/cultural_lora/smoke/adapter/` |
| 复检 | `smoke_fitcheck`（200 step）：无「答案截光」跳过，拟合逻辑 OK |

说明：首轮冒烟曾因尾部截断挤掉 MSVD 短答；已改为**减帧/降像素重编码、保留答案**。正式全量请用当前代码。

### 1.3 尚未产出（全量后补）

| 实验 | 状态 |
|------|------|
| Cultural LoRA 全量 1 epoch | 未跑 |
| OmniVDCBench `heritage_zh`：Base vs LoRA | 未跑 |
| Video-MME overall：Base vs LoRA | 未跑（Base 已有 63.7%） |

---

## 2. 实现技术

| 项 | 选择 |
|----|------|
| 基座 | 本地 `VQA/mage_vl/models`（`microsoft/Mage-VL`，`local_files_only`，不复制第二份） |
| 框架 | PyTorch + PEFT LoRA；自建 train loop（非 HF Trainer） |
| Processor | `AutoProcessor` / `AutoModelForCausalLM`，`trust_remote_code` |
| 存盘 | 只存 `adapter/`；训练结束最多保留 1 个 ckpt 再收敛为 `adapter/` |
| 显存 | gradient checkpointing；`use_cache=False`；超长则均匀减帧 / 降 `max_pixels`，**不裁答案** |
| 入口 | `VQA/mage_vl/cultural_lora/train_lora.py` |
| 环境 | `/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/python`（缺包用 `uv pip install --python ...`） |

---

## 3. 微调策略与固定权重

| 项 | 设定 |
|----|------|
| **冻结** | 全部视觉侧：`model.visual`（Mage-ViT）及非 LoRA 参数 |
| **可训** | 仅 LLM（Qwen3-4B）上 LoRA：`q/k/v/o_proj` + `gate/up/down_proj` |
| LoRA | r=16，alpha=32，dropout=0.05 |
| 优化 | AdamW，lr=1e-4；batch=1；grad_accum=8（有效 batch≈8） |
| 时长 | **1 epoch**（全量）；冒烟用 `max_steps` |
| 叙事 | 域适配，**不是**新视觉骨干 |

可训参数量级（冒烟实测）：约 **33.0M / 4.77B（0.69%）**。

---

## 4. 微调数据集

| 来源 | 用量 | 角色 | 落地路径 |
|------|------|------|----------|
| MSVD-QA | **8k** QA（1053 视频，帧包） | 稳住视频问答 | `VQA/mage_vl/data/MSVD-QA/train_8k/` |
| CiQi-VQA | **16k**（20533 图） | 文物/瓷器术语与属性 | `VQA/mage_vl/data/CiQi-VQA/train_16k.jsonl` + `images/` |

- 合计 **24k** 样本；seed=42；原全量 jsonl / parquet 已硬删。  
- **不用** ActivityNet；ChinaHeritaQA **不进训练**。  
- CiQi 答案训练时截到约 1024 字（长文）；MSVD 短答完整保留。

评测（训后，零样本）：OmniVDCBench `heritage_zh`（主）；Video-MME overall（对照，已有 Base）。

---

## 5. 阶段

```text
[已完成] 子集硬抽硬删 + verify(subset/after_delete)
[已完成] 冒烟 1k step + verify(after_smoke)；fitcheck 200 step
[下一步] 全量 1 epoch → 写出正式 adapter
[P0]     OmniVDCBench heritage_zh：Base vs LoRA
[P0]     Video-MME overall：LoRA 复测（对照 Base 63.7%）
[其后]   Oracle 位置 / codec 效率表（见原方案意图；不挡 LoRA 主线）
```

全量命令示例：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/python VQA/mage_vl/cultural_lora/train_lora.py \
  --model VQA/mage_vl/models \
  --max-steps 24000 \
  --output-dir VQA/mage_vl/outputs/cultural_lora/full_1ep
```

（24k≈1 epoch 按「每样本 1 step」计；可按墙钟再调。）

---

## 6. 磁盘与 4090 时间

### 6.1 磁盘（本机实测，硬删后）

| 项 | 约占用 |
|----|--------|
| Mage-VL 基座权重 | **9.9 G** |
| MSVD `train_8k/` | **8.2 G** |
| CiQi 16k 子集 | **0.4 G** |
| LoRA adapter（冒烟） | **0.13 G** |
| 训练数据+权重合计 | **≈18.6 G** |
| 建议空闲余量（ckpt/缓存） | **≥5 G** |
| 当前数据盘余量（参考） | ≈20 G free / 140 G |

省盘原则：只存 adapter；不 `snapshot_download` 第二份 Mage；CiQi/MSVD 只留子集。

### 6.2 一张 RTX 4090 24GB 时间

| 阶段 | 墙钟（实测/估算） |
|------|-------------------|
| 冒烟 1k step | **≈11.5 min**（实测 ≈1.4–1.5 it/s） |
| 全量 1 epoch（≈24k step） | **≈5–12 h**（按冒烟速率外推；长序列偏慢取上限） |
| 原计划经验区间 | 1–2 天（偏保守；本机冒烟后更接近 **半天内**） |

另计：OmniVDC / Video-MME 评测各数小时到一日级（视子集与 codec）。

---

## 7. 相关路径

| 路径 | 说明 |
|------|------|
| `VQA/mage_vl/cultural_lora/` | 准备 / 训练 / 校验 / README |
| `VQA/mage_vl/outputs/cultural_lora/smoke/adapter/` | **冒烟实验结果（adapter）** |
| `VQA/mage_vl/models/` | 基座（只读） |
| `VQA/eval_results/Video-MME.md` | Base 63.7% |
| `VQA/envs/requirements-cultural-lora.txt` | peft / duckdb 等 |
