# Mage Query-IP 选区 MVP（本目录）

对齐 [`docs/PLAN_mage_query_IP_select.md`](../../docs/PLAN_mage_query_IP_select.md) **§0 / §5.5**。

## 做什么

```text
Mage codec（cv-preinfer 出 canvas + src_positions）
  → 问题类型 static/dynamic（关键词）
  → 同 K 四路选区：
       prefix / random
       queryip          = α·motion_proxy + β·residual_proxy
       queryip_bitcost  = α·official_bitcost + β·0
  → 稠密打包 canvas → Mage-ViT + Qwen3 解码
```

- **不重训**；**不改** `VQA/mage_vl/` 源码（仅 import + 包装）。
- **不做** QuoTA / FlexSelect / 原 ETS LLaVA 评测；路线 A 真 MV **已归档**。
- 完整上游 Explore-Then-Select 树在仓外：`/root/autodl-tmp/Explore-Then-Select`。

## 目录

| 路径 | 说明 |
|------|------|
| `mage_query_ip/question_type.py` | 静/动分类 + α/β 表 |
| `mage_query_ip/codec_reweight.py` | canvas 旧路径 + patch 级 **proxy** 同 K |
| `mage_query_ip/mvres_score.py` | 官方 **bitcost** 打分 / `select_patches_bitcost` |
| `mage_query_ip/mage_forward.py` | Mage 加载；`encode_codec` / `run_select_mode`（四模式） |
| `mage_query_ip/smoke_soccer_samek.py` | soccer 同 K **四路** → `SAMEK4_*` + json |
| `mage_query_ip/eval_short60_samek.py` | short 前 60 题四路 Acc → samek4 jsonl |
| `scripts/smoke_soccer_samek.sh` | 四路入口（Agent 不代跑） |
| `COMMANDS.md` | **你要执行的全部终端命令** |

评测适配（VLMEvalKit）：模型名 `Mage-VL-QueryIP`，见 `VQA/VLMEvalKit/vlmeval/vlm/mage_vl_query_ip.py`（全量行 B 仍为 canvas proxy）。

## 同 K 成功标准（soccer 四路）

1. 四路 `n_vis_tokens` / `n_patches` 相同。
2. `queryip_bitcost` kept 与 `queryip` Jaccard **&lt; 1**（有差异）。
3. 答案非空 → `SAMEK4_OK`；完全重合 → `SAMEK4_REDUNDANT`。

**状态：** 代码已就绪；GPU 结果待你跑 §COMMANDS §2。

## 诚实边界

- `queryip`：解码 canvas 上的帧差/方差 **proxy**，不是 codec 内部 MV。
- `queryip_bitcost`：官方 block **bitcost** 单通道（无残差图）；论文可写「官方 codec bitcost + 问题重加权」，**不能**写「真 MV+残差」。
