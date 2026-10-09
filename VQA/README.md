# 工作 B · 视频问答 / 导览（占位）

本目录预留 **问答与评测** 工作的接入位置。第一阶段系统仓以稀疏 3DGS 重建为主；问答代码尚未迁入。

## 当前代码位置（仓库外）

| 项目 | 本机相对路径（示例） | 用途 |
|------|----------------------|------|
| Mage | `../../Mage` 或 `../../Mage_ViT` | 视频/图像理解与生成相关 |
| VLMEvalKit | `../../VLMEvalKit` | 视觉语言模型评测 |

环境依赖 freeze（已进本仓，供后续复现）：

- [`../envs/requirements-mage-vit.txt`](../envs/requirements-mage-vit.txt)
- [`../envs/requirements-vlmevalkit.txt`](../envs/requirements-vlmevalkit.txt)

## 后续接入方式（待定）

1. 将 Mage / VLMEvalKit **迁入**本目录（或作为 git submodule），或  
2. 保持仓外独立仓库，本目录只保留接口文档与启动脚本。

与工作 A 的衔接见 [`../docs/system/README.md`](../docs/system/README.md)。

## 本目录约定

- 暂不放训练数据与大权重（沿用根 `.gitignore`）  
- 迁入代码前保持本 README，避免空目录无法被 git 跟踪时丢失占位说明  
