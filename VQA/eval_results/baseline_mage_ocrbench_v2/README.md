# baseline_mage_ocrbench_v2 · 原版 Mage-VL × OCRBench_v2

> 状态：**进行中 / 已暂停保全**（Agent 不代跑）  
> 命令：[`COMMANDS.md` §9 / §9.1](../../Explore-Then-Select/COMMANDS.md)

## 已保全进度（勿删）

| 路径 | 说明 |
|------|------|
| `Mage-VL/T20261027-105443/01_OCRBench_v2.pkl` | 首跑 ~500 |
| `Mage-VL/T20261027-112350/01_OCRBench_v2.pkl` | **最新续跑** ~1180（reuse 后累计） |
| `_backup_pkl/01_OCRBench_v2_n1180_final.pkl` | 停卡前备份 |

续跑**必须** `--reuse --reuse-aux all`，**不要** `--verbose`（只要进度条）。

## 静默续跑（复制）

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source /root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/activate
REPO=/root/autodl-tmp/Sparse-3DGS-heritage-VQA
export LMUData="$REPO/VQA/datas/lmu"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/VQA/VLMEvalKit:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKIP_ERR=1

nohup python "$REPO/VQA/VLMEvalKit/run.py" \
  --model Mage-VL --data OCRBench_v2 \
  --work-dir "$REPO/VQA/eval_results/baseline_mage_ocrbench_v2" \
  --reuse --reuse-aux all \
  > "$REPO/VQA/eval_results/logs/mage_ocrbench_v2_quiet_resume.out" 2>&1 &

tail -f "$REPO/VQA/eval_results/logs/mage_ocrbench_v2_quiet_resume.out" | grep --line-buffered 'Infer Mage-VL'
```

## 指标

跑完：English / Chinese Overall Score（VLMEvalKit `OCRBench_v2.evaluate`）。

## 盘

- TSV/图缓存：`$LMUData` = `VQA/datas/lmu`
- 本目录：预测 pkl/xlsx/score + README
