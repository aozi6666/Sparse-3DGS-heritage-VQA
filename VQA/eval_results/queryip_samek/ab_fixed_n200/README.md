# α/β 三角对照（E2 · 已归档 2026-10-23）

- n=200 · `keep_ratio=0.75` · `seed=0` · `partial=false`
- 计划：[`docs/PLAN_mage_query_IP_select.md`](../../../../docs/PLAN_mage_query_IP_select.md) §2.6 · §4.4.3

| mode | Acc | correct |
|------|-----|---------|
| queryip | 63.5% | 127/200 |
| queryip_dyn | 63.5% | 127/200 |
| queryip_sta | 64.0% | 128/200 |

**结论：** Static/Dynamic α/β 自适应 **不作贡献**；保留 `queryip` 作 proxy 选区。勿删本目录。
