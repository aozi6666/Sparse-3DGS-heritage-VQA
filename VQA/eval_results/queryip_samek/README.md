# 同 K 消融产物

`keep_ratio=0.75`（除非文件名带 `krXXX`）。四路：`prefix` / `random` / `queryip` / `queryip_bitcost`。

## n200（主表）

| 文件 | 含义 |
|------|------|
| `n200/..._n200_purebitcost.*` | miss→0；bitcost **58.5%**，proxy **64.0%** |
| `n200/..._n200_hybrid.*` | miss→proxy；bitcost **58.0%**，proxy **64.0%** |

## n60

| 文件 | 含义 |
|------|------|
| `n60/..._short60_samek.jsonl` | 早期三路 proxy |
| `n60/..._kr075.*` | 四路 n=60（bitcost=proxy=61.7） |

## ksweep

仅完成 0.50 / 0.60；0.75 在 n60；0.90 未跑。`ksweep.csv` 可能不完整。
