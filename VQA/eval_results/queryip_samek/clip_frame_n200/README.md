# CLIP 帧预算同 K（E0 / E1 / E4 · 已归档 2026-10-23）

- n=200 · `keep_ratio=0.75` · `seed=0` · `partial=false`
- 计划：[`docs/PLAN_mage_query_IP_select.md`](../../../../docs/PLAN_mage_query_IP_select.md) §2.5

| mode | Acc | correct |
|------|-----|---------|
| random (E1) | 59.5% | 119/200 |
| **queryip (E0) · 保留** | **63.5%** | 127/200 |
| queryip_clip (E4) | **58.5%** | 117/200 |

Δ(queryip−random)=**+4.0 pp**（点估计；McNemar 见计划 §8.1）。E4 失败消融；不做方案 B。勿删本目录。
