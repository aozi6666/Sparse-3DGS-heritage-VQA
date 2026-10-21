# 路线 A「真 MV / 残差」— 已归档（短注）

> **状态：已归档（2026-10-21）· 不再实施**  
> 主计划与定稿结果见 [`PLAN_mage_query_IP_select.md`](PLAN_mage_query_IP_select.md)。

## 为何停

开放评测栈（`codec-video-prep` / `cv_reader_fast`）**只有官方 bitcost 单通道**，没有可分开加权的 MV + 残差。  
「改 `cv-preinfer`、把 bitcost 当 MV+残差」不合理；即便读出两路，第二刀也只在 codec 已留 canvas 上砍 K，**不能**拉回满预算 Acc。

## 后来实际做了什么

| 路径 | 结果 |
|------|------|
| 像素 proxy + 问题 α/β（`queryip`） | **主推** · n=200 同 K 64.0% > random 59.5% |
| 官方 bitcost / hybrid | **失败消融** · 均低于 random |
| 真 MV+残差窄实验 | **未做、不做** |

## 明确不做

- 改 `cv-preinfer` / site-packages 伪装真 MV  
- 为「看起来用了真分」重跑全量 Video-MME  
- 论文写「使用了 codec 运动向量与残差能量」
