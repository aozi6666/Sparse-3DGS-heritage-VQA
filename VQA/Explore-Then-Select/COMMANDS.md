# Mage Query-IP — 请你本地执行的命令（Agent 不代跑评测）

工作目录默认：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA`  
分支要求：`Mage_LoRA`

本仓 `VQA/Explore-Then-Select/` **只保留** `mage_query_ip/` + 冒烟/同 K 脚本。完整上游 ETS 在仓外：`/root/autodl-tmp/Explore-Then-Select`。

---

## 0. 准备

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
git branch --show-current   # 期望：Mage_LoRA

export PATH="/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin:$PATH"
nvidia-smi
/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.__version__)"
ls VQA/mage_vl/models/config.json
ls VQA/mage_vl/models/examples/soccer-broadcast.mp4
```

---

## 1. 已归档：行 B 全量 Video-MME（勿重跑除非必要）

结果：[`VQA/eval_results/Mage-VL-QueryIP_Video-MME.md`](../eval_results/Mage-VL-QueryIP_Video-MME.md)  
overall **62.3%**（vs 行 A 63.7%）；管线可行，Acc 未涨。

---

## 2. 路线 B · Soccer 同 K 三路（你现在跑这个）

一次 codec，三路 `prefix` / `random` / `queryip`，`keep_ratio=0.75`。  
成功：`SAMEK_OK`（同 token；QueryIP kept 异于随机/原序；答案非空）。

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
chmod +x VQA/Explore-Then-Select/scripts/smoke_soccer_samek.sh
./VQA/Explore-Then-Select/scripts/smoke_soccer_samek.sh
```

可选环境变量：`KEEP_RATIO` `SEED` `NUM_FRAMES` `MAX_PIXELS` `MAX_NEW_TOKENS` `MODEL` `VIDEO` `VENV_BIN`。

---

## 3. 路线 B · Video-MME short 前 60 题（仅 soccer 出现 SAMEK_OK 后再跑）

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
chmod +x VQA/Explore-Then-Select/scripts/eval_short60_samek.sh
./VQA/Explore-Then-Select/scripts/eval_short60_samek.sh
```

结果写入：`VQA/eval_results/Mage-VL-QueryIP_short60_samek.jsonl`，终端打印三路 Acc。  
可选：`N=60` `KEEP_RATIO=0.75` `SEED=0`。

---

## 4. 旧命令（参考）

原 canvas 级 soccer 冒烟 / 全量行 B：

```bash
./VQA/Explore-Then-Select/scripts/smoke_soccer.sh
bash VQA/VLMEvalKit/sh/run_mage_queryip_videomme.sh
```

---

## 5. 不做（本阶段）

- 路线 A（改 `cv-preinfer` 导出真分）——需你确认源码/可改二进制后再开  
- QuoTA、FlexSelect、Cultural LoRA 全量、重跑 714 题
