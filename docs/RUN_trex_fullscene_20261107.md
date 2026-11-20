# trex 全视图、真实背景重建流程

更新：2026-11-07。原模式为默认，新模式需显式使用 `--full_scene` 或本页专用入口。

## 目标与方法范围

- 所有55张照片参与训练，保留恐龙骨架和展厅真实背景。
- 自动完成相机去畸变、图片命名兼容、全图有效区域、VGGT辅助深度、全场景点云、GS+Drop、LOO、LoRA、扩散修复和七张同视角渲染。
- 全场景初始化为「原SfM背景点＋VGGT可靠深度补点」，不把全白有效区域当作对象轮廓，不声称本次采用Visual Hull的已发表三模块原版SynGS。
- 旧 `sh/lineB_gaussianobject` 入口、原Visual Hull／SAM对象模式的默认行为保持不变。

## 掩膜与辅助深度

工程原有 `segment_anything.ipynb` 和 `segment_anything_2.ipynb` 使用SAM与提示点生成对象轮廓；生成的真实对象掩膜可用于原去背景路线，需相应SAM权重和提示点。

本次全背景路线自动生成「所有像素有效」掩膜：它用于统一读入接口，保持整幅RGB，不代表已经分割出恐龙；粗训练与repair关闭轮廓损失、随机替换背景。

辅助深度复用已有 `3DGS/vggt/models/model.pt`，逐帧预测并用当前场景COLMAP观测对齐尺度。深度保存在 `zoe_depth/*.npy`，目录名为历史接口兼容，真实来源在JSON中注明为VGGT，不是Zoe或真值深度。浮点存储避免COLMAP大尺度深度溢出16位PNG；旧PNG仍能读取。

## 启动

在AutoDL终端执行：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
bash 3DGS/GaussianObject/sh/full_scene/run_trex.sh
```

只查看方案（不准备数据、不训练）：

```bash
bash 3DGS/GaussianObject/sh/full_scene/run_trex.sh --plan
```

只运行CPU预处理：

```bash
bash 3DGS/GaussianObject/sh/full_scene/run_trex.sh --prepare-only
```

检查GPU、数据与权重：

```bash
bash 3DGS/GaussianObject/sh/full_scene/run_trex.sh --check
```

默认分辨率因子4，粗训练30000步、LOO每视图10000步、LoRA1800步、repair4000步。LOO仍包含全部55个视图，整体训练量明显高于旧4／9视图流程。先检查资源报告，实际耗时以日志为准。完整模型加载要求至少12 GiB CPU内存；当前无卡实例为2 GiB，GPU实例需有相应内存。

## 路径与结果

| 内容 | 路径（相对GaussianObject） |
|---|---|
| 原始资料，只读 | `data/LLFF/trex` |
| 去畸变及准备资料 | `data/LLFF/trex_fullscene` |
| 运行根目录 | `output/full_scene/trex_fullscene_v1` |
| 粗训练模型 | `.../coarse/point_cloud/iteration_*` |
| 最终精化模型 | `.../refinement/final/save/last.ply` |
| 七张单独渲染 | `.../display/01.png` 至 `07.png` |
| 同视角拼图 | `.../display/comparison_1_to_7.png` |
| 原始参考照片 | `.../display/reference.png` |
| 图与模型对应 | `.../display/render_manifest.json` |
| 命令、阶段耗时、日志 | `.../logs` 与 `.../stages` |

同一相机默认选择排序后中间帧。可以通过 `--camera-name DJI_20200223_163619_411` 指定原始相机名（以 `--plan` 输出和资料清单为准）。固定相机、内外参、背景模式、尺寸；不靠模糊、破坏图片制造差异。

六个粗模型检查点为500／1000／3000／7000／15000／30000步，第七个为最终repair。根据同一训练参考图的PSNR降序编号1—7，实际阶段写入清单；最终模型未必排名第一，不人为调整排序。这是全视图训练后的阶段展示，不是独立测试视图的定量比较或七种算法结果。

## 续跑与磁盘

重复相同命令会验证已准备数据，并跳过有完成记录的阶段。修改主要配置或代码时需用新 `--run-tag`，避免混合实验。发生错误后从明确阶段续跑，例如：

```bash
bash 3DGS/GaussianObject/sh/full_scene/run_trex.sh --start lora
```

LOO按一个视图一对阶段执行。只有该新任务的样本和 `diffs.pkl` 已生成后，才清理该视图的临时检查点和临时模型；原始资料、旧实验、七张展示所需粗模型与最终模型均保留。未完成的训练阶段可能重跑，不声称支持任意迭代位点无损续训。

准备数据先写独立临时目录，完成校验后切换到目标目录。已存在且来源不匹配的目标目录会拒绝覆盖。相机从SIMPLE_RADIAL真实去畸变成PINHOLE／SIMPLE_PINHOLE，缩略图与相机文件名一致；RGB、相机、掩膜、划分清单写入来源指纹。为适应2 GiB CPU实例，采用OpenCV逐图去畸变，并用pycolmap同步写回针孔内参及二维观测坐标，保持三维点和位姿坐标系不变。

## 验证与限制

本次已完成55图CPU准备、命名、深度单位、旧默认值及入口检查。去畸变后的二维观测坐标与预期变换一致（本次抽查误差为0），原55张图像及原COLMAP文件SHA-256未变化；读取真实trex背景与旧kitchen对象掩膜均通过。6项CPU兼容测试通过。GPU不可用期间不执行正式训练；最终七张图必须等待真实训练完成后生成。完整GPU数值行为和修复效果以之后实际运行日志为准。

CPU检查命令：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject
python -m unittest discover -s tests -p test_full_scene_cpu.py -v
```
