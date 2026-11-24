# trex 4/6/9视角：基础3DGS与VGGT＋Drop的受控对照

日期：2026-11-09。用户要求提供终端命令，由用户启动，本轮不代跑GPU训练。

## 比较什么

已完成组合：`output/trex_sparse469_v1/{v4,v6,v9}`，预训练VGGT补点＋DropGaussian。

新增基线：`output/trex_sparse469_baseline_v1/{v4,v6,v9}`，关闭VGGT补点、关闭DropGaussian，不载入VGGT深度，不计算深度监督。

两者沿用同一工程训练器、相机标定、训练/测试划分、基础三角化点云、分辨率、随机种子初始化方式和优化设置。新命令直接从已成功完成组合的 `.train.done.json` / `.evaluate.done.json` 派生：只更换数据和输出路径、将初始化设置为 `origin`（各组基础三角化点云）、移除 `--use_drop`、`--drop_rate`、`--drop_schedule`。原有组合的 `--mono_depth_weight 0` 保持不变。

本入口是**本工程内的基础3DGS对照**，不是官方3DGS仓库与其原始默认设置的独立复现。GaussianObject工程中的其他训练设置，例如增密、离群点剔除和透明度重置保持与组合实验相同，目的是测量增添“VGGT＋Drop”的合并效果。若论文要声称“与官方3DGS复现对比”，应另建官方实现和完整协议。

当前两项处理同时移除，结果只能说明二者的合并贡献；要分开归因，还需“仅VGGT”和“仅Drop”两组。此流程也不包含发表版SynGS的Visual Hull、LoRA或repair。

## 固定条件

| 项目 | 设置 |
|---|---|
| 4视角训练索引 | 1, 21, 34, 54 |
| 6视角训练索引 | 1, 14, 21, 34, 41, 54 |
| 9视角训练索引 | 1, 7, 14, 21, 28, 34, 41, 47, 54 |
| 公共测试索引 | 0, 8, 16, 24, 32, 40, 48 |
| 基础点数 | 4视角4435；6视角6416；9视角9447 |
| 每组步数 | 10000 |
| 输入分辨率 | 1008×756 |
| 背景 | 完整真实背景 |
| 增密终止步数 | 6000 |
| 高斯数量软阈值 | 500000 |
| 深度损失权重 | 0 |
| Drop | 关闭，不传`--use_drop` |
| 指标 | 共同7张测试图的PSNR / SSIM / LPIPS平均值 |

索引从0开始，按图像stem排序。沿用全量COLMAP提供的相机标定与特征轨迹身份；基础点坐标和颜色来自对应训练视角的重新三角化，不复用55视角的场景点云。新基线通过原VGGT推理时保存的深度sidecar内`geometry_sha256`核对原始基础几何未变，并逐项检查PLY坐标/颜色与NPZ一致。

## 启动命令

在远程终端执行：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
mkdir -p 3DGS/GaussianObject/output
nohup bash 3DGS/GaussianObject/sh/full_scene/run_trex_sparse469_baseline.sh > 3DGS/GaussianObject/output/trex_sparse469_baseline_v1.console.log 2>&1 < /dev/null &
tail -f 3DGS/GaussianObject/output/trex_sparse469_baseline_v1.console.log
```

三组依次运行。Ctrl+C仅退出tail，后台任务继续。相同输出有文件锁防止重复并行执行；成功阶段记录done标记，重跑命令会跳过完整阶段。未完成的粗训会从该组起点重跑，因为没有保存优化器快照。参数、代码、原组合结果或输入改变时会要求新`--run-tag`，避免混用。

完成标记：新结果根目录存在`COMPLETE.json`，日志尾部打印`COMPLETE:`。

## 输出位置

`/root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/output/trex_sparse469_baseline_v1/`

- `v4/`、`v6/`、`v9/`：各组最终基础模型、训练和评测日志、7张测试图的GT及渲染结果、与组合的逐视角拼图`render_comparison.png`。
- `baseline_vs_vggt_drop.md`：基础3DGS与已有组合的指标表，以及“组合减基础”的有符号差值。
- `baseline_vs_vggt_drop.csv`：两方法、三视角数量的指标数据。
- `metrics_comparison.png`：PSNR / SSIM / LPIPS对比曲线，两条曲线对应两方法。
- `render_comparison.png`：三个视角数量下，同一固定测试相机的GT / 基础3DGS / VGGT＋Drop拼图；固定使用第一个测试相机，不按效果挑图。
- `run_config.json`：派生后的真实命令、来源和代码指纹。

汇总在每完成一组后更新，`COMPLETE.json`出现后才是三组完整结果。

```bash
cat /root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/output/trex_sparse469_baseline_v1/baseline_vs_vggt_drop.md
```

数据在`data/LLFF/trex_sparse469_baseline_v1/{v4,v6,v9}`。原图、相机文件和基础点云采用输入链接，不复制照片；不链接`zoe_depth`、置信度或VGGT增强点云。只保存每组最终模型，不保存中间模型、视频和优化器快照；成功粗训后删除本次新输出里可再生成的`input.ply`。已有组合不重训、不覆盖。

## 验证范围

已完成CPU准备、Python编译及Bash语法检查。已用实际数据读取器检查三个基线分别加载4/6/9张训练图和7张公共测试图，加载的是4435/6416/9447个基础点；检查初始化为origin，无VGGT深度目录、无Drop参数，其他CLI控制与原组合逐项一致。

本轮未启动GPU训练或评测；不能提前断言组合优于基础3DGS。以实际新生成的指标差值和渲染为准。

新增入口：`3DGS/GaussianObject/tools/run_sparse_469_baseline.py`、`3DGS/GaussianObject/sh/full_scene/run_trex_sparse469_baseline.sh`。既有训练器、初始化器、渲染器和组合入口均未修改。
