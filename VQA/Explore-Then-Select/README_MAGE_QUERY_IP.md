# Mage Query-IP 选区 MVP（本目录）

对齐 [`docs/PLAN_mage_query_IP_select.md`](../../docs/PLAN_mage_query_IP_select.md) **§0 简单版**。

## 做什么

```text
Mage codec（cv-preinfer 出 canvas + src_positions）
  → 问题类型 static/dynamic（关键词）
  → score' = α(q)·motion_proxy + β(q)·residual_proxy（canvas 帧差/方差）
  → 固定预算 K 选留 canvas（保持 grid 合法）
  → Mage-ViT + Qwen3 解码
```

- **不重训**；**不改** `VQA/mage_vl/` 源码（仅 import + 包装）。
- **不做** QuoTA / FlexSelect / 原 ETS LLaVA 评测。
- 完整上游 Explore-Then-Select 树在仓外：`/root/autodl-tmp/Explore-Then-Select`（本仓不保留 `model/` `test/` 等）。

## 目录

| 路径 | 说明 |
|------|------|
| `mage_query_ip/question_type.py` | 静/动分类 + α/β 表 |
| `mage_query_ip/codec_reweight.py` | canvas 旧路径 + **patch 级** proxy 同 K（prefix/random/queryip） |
| `mage_query_ip/mage_forward.py` | Mage 加载；`encode_codec` / `run_select_mode` |
| `mage_query_ip/smoke_soccer.py` | 旧 canvas 冒烟 |
| `mage_query_ip/smoke_soccer_samek.py` | **路线 B** soccer 同 K 三路 |
| `mage_query_ip/eval_short60_samek.py` | short 前 60 题同 K Acc |
| `scripts/smoke_soccer_samek.sh` | 同 K 入口（Agent 不代跑） |
| `COMMANDS.md` | **你要执行的全部终端命令** |

评测适配（VLMEvalKit）：模型名 `Mage-VL-QueryIP`，见 `VQA/VLMEvalKit/vlmeval/vlm/mage_vl_query_ip.py`。

## 冒烟成功标准

1. default / static-reweight / dynamic-reweight 均能出回答。
2. **同一 `budget_k`** 下，static 与 dynamic 的 `kept_canvas_indices` 或 `mean_motion_kept` 有可见差异。
3. 不要求 Video-MME 涨点。

**状态（2026-10-21）：** 已满足 → `SMOKE_OK`（见计划 §0.1）。

## 诚实边界

Proxy 来自解码 canvas，**不是** cv-preinfer 内部真 MV/残差图；论文叙事写「codec 输出 + 问题条件化重加权」。
