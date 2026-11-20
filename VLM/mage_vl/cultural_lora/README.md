# Cultural LoRA（Mage_LoRA 分支）

冻 Mage-ViT，仅 LLM 侧 PEFT LoRA。数据：MSVD **8k QA** + CiQi **16k**（硬抽硬删）。默认只跑 **~1k step 冒烟**，全量 1 epoch 另开。

## 约束

- 权重只用本地 `VQA/mage_vl/models/`（`local_files_only`，不 `snapshot_download`）
- 只存 PEFT adapter；训练结束仅保留 `outputs/.../adapter/`
- CiQi / MSVD 子集先建再 `--confirm-delete`（不可逆）

## 0. 分支与依赖

```bash
cd ~/autodl-tmp/Sparse-3DGS-heritage-VQA
git branch --show-current   # 必须 Mage_LoRA

# 在你的训练 venv 里（示例）
pip install -r VQA/envs/requirements-cultural-lora.txt
# 还需已装好可跑 Mage-VL 推理的 torch / transformers（见 mage_vl/requirements.txt）
```

## 1. 抽子集（先不删）

```bash
cd ~/autodl-tmp/Sparse-3DGS-heritage-VQA

python VQA/mage_vl/cultural_lora/prepare_ciqi_subset.py --n 16000 --seed 42
python VQA/mage_vl/cultural_lora/prepare_msvd_subset.py --n 8000 --seed 42 --max-frames 16

python VQA/mage_vl/cultural_lora/verify_disk_policy.py --stage subset
```

说明：

- CiQi：`train_16k.jsonl` + `subset_manifest_16k.json`（本阶段固定 seed 随机；仓内无 CiQi-Bench 排除表）
- MSVD：展开嵌套 `qa` 后抽 8k，写入 `train_8k/{qa.jsonl,frames/*.npy}`。当前盘上 `train-00000-of-00011.parquet` 截断不可读，脚本会自动跳过；其余 shard 可展开约 2.8 万 QA，足够抽 8k

## 2. 确认后硬删

```bash
# 看完 verify 再执行；不可逆
python VQA/mage_vl/cultural_lora/prepare_ciqi_subset.py --n 16000 --seed 42 --confirm-delete --delete-full-backup
python VQA/mage_vl/cultural_lora/prepare_msvd_subset.py --n 8000 --seed 42 --confirm-delete

python VQA/mage_vl/cultural_lora/verify_disk_policy.py --stage after_delete
```

- CiQi：删未引用图；`train.jsonl` 换成 16k；`--delete-full-backup` 再删 `train.jsonl.full.bak`
- MSVD：删全部原始 `train-*-of-*.parquet`，保留 `train_8k/`

## 3. 清 HF 重复缓存（不碰本地 Mage 权重）

```bash
# 只查看体积，确认没有第二份 Mage-VL 权重后再删
du -sh ~/.cache/huggingface/hub 2>/dev/null
ls ~/.cache/huggingface/hub 2>/dev/null | head

# 若存在 models--microsoft--Mage-VL 且你确认只用 VQA/mage_vl/models：
# rm -rf ~/.cache/huggingface/hub/models--microsoft--Mage-VL

# 切勿删除：
#   ~/autodl-tmp/Sparse-3DGS-heritage-VQA/VQA/mage_vl/models
```

## 4. 冒烟（~1k step，你手动跑）

显存：默认开 **gradient checkpointing**、CiQi 答案截到 **1024 字**、`max_seq_len=3072`。  
序列超长时会**均匀减帧 / 降 max_pixels 后重编码**，**绝不从尾部裁掉答案**（避免 MSVD 短答被视频 token 挤没）。请用 venv 的 python（无 pip 时用 `uv pip install --python ... peft duckdb`）。

```bash
cd ~/autodl-tmp/Sparse-3DGS-heritage-VQA

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/python VQA/mage_vl/cultural_lora/train_lora.py \
  --model VQA/mage_vl/models \
  --max-steps 1000 \
  --grad-accum 8 \
  --max-frames 8 \
  --max-pixels 150000 \
  --max-answer-chars 1024 \
  --max-seq-len 3072 \
  --save-every 500 \
  --output-dir VQA/mage_vl/outputs/cultural_lora/smoke

/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/python VQA/mage_vl/cultural_lora/verify_disk_policy.py \
  --stage after_smoke \
  --output-dir VQA/mage_vl/outputs/cultural_lora/smoke
```

若仍偶发 OOM：加 `--ciqi-max-images 1` 或 `--max-pixels 100000`（脚本会 skip 单条 OOM 并继续）。

成功时 `smoke/adapter/` 下应有 `adapter_config.json` + `adapter_model.safetensors`，且无 `model-*-of-*.safetensors`。

## 5. 全量 1 epoch（本次不做；冒烟通过后再跑）

把 `--max-steps` 调大到覆盖约 24k 样本 × 1 epoch（按 `grad-accum` 换算），`--output-dir` 换新目录即可。

## 产物路径

| 路径 | 说明 |
|------|------|
| `VQA/mage_vl/data/CiQi-VQA/train_16k.jsonl` | CiQi 子集 |
| `VQA/mage_vl/data/MSVD-QA/train_8k/` | MSVD 紧凑子集 |
| `VQA/mage_vl/outputs/cultural_lora/smoke/adapter/` | 冒烟 adapter |
| `VQA/mage_vl/models/` | 基座（只读，勿复制） |
