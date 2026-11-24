# 三维重建工作区

从 reconstruction 开始，重建流程基于 GaussianObject。稀疏视角训练在这个工程中完成。DropGaussian_release 是用于核对算法的独立上游子模块，vggt 提供深度预测代码和模型。

| 要做的事 | 从哪里开始 |
|---|---|
| VGGT 与视觉外壳初始化、粗训、可选修复 | [主流程](reconstruction/workflows/README.md)，脚本按步骤编号 |
| 带背景的场景重建、稀疏视角对照 | [全场景入口](reconstruction/workflows/full_scene/) |
| 比较 Drop 与基础模型的增密梯度 | [梯度实验](reconstruction/experiments/gradient_depth_drop/) |
| 检查 VGGT 补点的贡献 | [horns 消融](reconstruction/experiments/horns_vggt_ablation/) |
| 生成论文渲染图 | [LLFF 场景](reconstruction/experiments/llff_scene_gallery/) 或 [Mip360 前景](reconstruction/experiments/mip360_foreground_gallery/) |
| 查历史指标与结论 | [评测归档](reports/eval_results/) |
| 查上游算法 | [DropGaussian](reconstruction/third_party/DropGaussian_release/README.md)、[VGGT](reconstruction/third_party/vggt/README.md) |

reconstruction/data 存放数据，output 存放实验产物。主流程调用工程根下的 train_gs.py、train_lora.py、train_repair.py 和 render.py。tools 保留数据准备与诊断脚本。workflows/_archive 保存旧启动脚本。


## 工程内部

| 目录或文件（相对于 reconstruction） | 用途 |
|---|---|
| workflows/、experiments/ | 完整训练流程和可单独运行的论文实验 |
| scene/、gaussian_renderer/、arguments/ | 场景读取、高斯表示、投影渲染与训练参数 |
| preprocess/、tools/、visual_hull.py、vggt_visual_hull_enhanced.py | 数据预处理、几何初始化与诊断 |
| cldm/、ldm/、train_lora.py、train_repair.py | 扩散模型、适配训练和修复监督 |
| third_party/DropGaussian_release/ | 原版 DropGaussian 对照实现，Git 子模块 |
| third_party/vggt/ | 上游 VGGT 及其 models/model.pt |
| models/、pretrained/ | 修复、分割等流程使用的本地权重及配置 |
| tests/ | 全场景兼容性测试，梯度分箱测试位于对应实验目录 |


运行环境位于仓库根目录的 .venv-ddgs-vggt。
