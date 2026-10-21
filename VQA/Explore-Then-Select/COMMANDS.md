# Mage Query-IP — 本地命令（Agent 不代跑）

工作目录：`/root/autodl-tmp/Sparse-3DGS-heritage-VQA` · 分支 `Mage_LoRA`  
计划：[`docs/PLAN_mage_query_IP_select.md`](../../docs/PLAN_mage_query_IP_select.md)  
产物：[`VQA/eval_results/README.md`](../eval_results/README.md) · 总览：[`VQA/README.md`](../README.md)

**定稿主推：`queryip`（proxy）。** bitcost / hybrid = 失败消融。

---

## 0. 准备

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
export PATH="/root/autodl-tmp/VLMEvalKit/vlmeval_env/bin:$PATH"
```

---

## 1. 已完成（勿重跑除非必要）

| 实验 | 结果位置 |
|------|----------|
| 行 A Mage 全量 | `eval_results/baseline_mage/` |
| 行 B QueryIP 全量（非同 K）62.3% | `eval_results/queryip_rowB_full/` |
| Soccer SAMEK4_OK | `eval_results/queryip_samek/soccer/` |
| short-60 | `eval_results/queryip_samek/n60/` |
| **n=200 主表**（proxy 64.0） | `eval_results/queryip_samek/n200/` |

---

## 2. proxy keep_ratio 扫描（已完成）

产物与做图 CSV：`eval_results/queryip_samek/ksweep_proxy/`（见该目录 README）。  
结论：0.3–0.75 queryip>random；0.9 略负。勿重跑除非改协议。

```bash
# 仅当需要重跑/续跑时：
nohup ./VQA/Explore-Then-Select/scripts/eval_proxy_ksweep.sh \
  > VQA/eval_results/logs/proxy_ksweep_nohup.out 2>&1 &
```

---

## 3. 后续预案：满 token Mage 效率曲线（先别跑）

见计划 **§5**。复用已有 ksweep 多 K，**只补满预算 Mage 一点** → `efficiency_curve/`。

---

## 4. 不做

- 同 K 全量 714 四路  
- 主推官方 bitcost  
- 改 `cv-preinfer` / 真 MV  
