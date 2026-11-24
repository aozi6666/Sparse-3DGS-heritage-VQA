# trex：当前收益与SynGS场景扩展三模块消融

## 已完成的事实

基础3DGS与VGGT＋Drop已在同样4/6/9训练划分、相同7测试图、10000步、1008×756完整真实背景设置下完成。结果来自真实JSON，没有改动或挑选渲染质量：

| 视角 | 配置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---:|---|---:|---:|---:|
| 4 | 工程基础3DGS | 15.7419 | 0.571614 | 0.474901 |
| 4 | 冻结VGGT＋Drop | 15.9338 | 0.583255 | 0.456844 |
| 6 | 工程基础3DGS | 16.8735 | 0.629745 | 0.440927 |
| 6 | 冻结VGGT＋Drop | 16.4759 | 0.624267 | 0.415625 |
| 9 | 工程基础3DGS | 17.9353 | 0.699587 | 0.401141 |
| 9 | 冻结VGGT＋Drop | 18.1150 | 0.701107 | 0.387629 |

4/9视角三个指标均有改善，但PSNR提升仅0.1920/0.1797dB；6视角PSNR下降0.3976dB、SSIM下降0.005478，LPIPS改善0.025302。现有结果显示部分收益，尚不支持“各设置全面优于基础3DGS”或单独归因某一模块。单场景、单随机种子没有稳定性或统计显著性证据。

## 用户本次确认的定义

1. 继续trex真实博物馆背景；Visual Hull只约束恐龙，背景另外保留。
2. 沿用当前单分支、有补偿的DropGaussian，暂不切换为本地论文正文的双分支无补偿DVR。
3. VGGT保持预训练冻结推理，不微调。

因此应称为**SynGS的真实背景场景扩展工程消融**。它不等于发表版SynGS的精确复现：本地CVMP正文还有适配的多视图深度分支、深度过滤与双分支DVR描述；当前VGGT缓存是冻结单帧推理，当前Drop实现也不同。不能把新增三模块全部开启后自动改名为“完整发表版SynGS”或用它预先保证收益。

## 消融矩阵

B表示给定相机标定与特征轨迹下的工程基础3DGS；H表示恐龙前景Visual Hull表面补点；V表示已缓存的冻结VGGT补点；D表示现有DropGaussian。

| 配置 | H | V | D | 状态 |
|---|---:|---:|---:|---|
| B | 0 | 0 | 0 | 已完成、复用 |
| D | 0 | 0 | 1 | 待训练 |
| V | 0 | 1 | 0 | 待训练 |
| VD | 0 | 1 | 1 | 已完成、复用 |
| H | 1 | 0 | 0 | 待前景轮廓可靠后训练 |
| HD | 1 | 0 | 1 | 同上 |
| HV | 1 | 1 | 0 | 同上 |
| HVD | 1 | 1 | 1 | 同上，完整工程组合 |

三个视角数量共24组，复用6组，新增18次粗训练。预计新增训练及评测约1–2小时，基于已实测基础组176.9/183.6/194.5秒、VD组241.3/288.2/274.7秒；新增VH可能增加高斯数量，最终耗时应以实际日志为准。掩膜人工修正时间不包括在这个GPU训练估算内。

逐阶段：B→H→HV→HVD。去模块：HVD−VD检验H、HVD−HD检验V、HVD−HV检验D。另报D−B、V−B、VD−V等，不只展示正向结果。模块作用可能依赖其他模块是否开启。

## 前景与背景怎样分开

- 所有配置都保留同一组训练视角重算的基础SfM点（包含背景），保留真实RGB训练损失，不使用抠图或白背景。
- H仅增加恐龙轮廓生成的表面点，绝不以全白有效掩膜冒充目标物体轮廓。
- H的包围盒只从训练SfM点的前景投影确定；候选体素需至少2个训练相机在画内约束，并与全部画内训练轮廓一致。裁出图像外的尾巴/脚部视为未知而非背景。轮廓膨胀默认3像素，网格144³，最多60000个H点。
- 由于LLFF主要是前向视角，Visual Hull可能受深度范围和包围盒截断影响；`hull_report.json`记录边界表面占比，不能把轮廓体当作真实三维GT。
- 4/6/9组各自只用自己的训练轮廓。测试轮廓放在单独`masks_eval`目录，构建H时不读取。
- V使用已有、同一训练集合生成的补点，完全复用，避免重新推理的随机差异。H/HD/HV/HVD使用同一固定H点集；V和VD也使用同一固定V点集。
- H与V的集合直接附加在同一基础点集上，使每个模块的新增点不随另一个模块开关而变化。不同点数带来的资源变化应与质量一起报告；后续若需排除“只是点更多”的解释，可另加等点数随机补点对照。

## 当前真正阻塞：恐龙轮廓不可靠

已从SAM官方固定提交安装独立vendor代码，并下载ViT-B约375MB权重，未升级或改动venv中的PyTorch。官方来源：https://github.com/facebookresearch/segment-anything ，提交`6fdee8f2727f4506cfbbe553e23b895e27956588`；权重SHA-256：`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`。

CPU实际试验发现，直接取最高SAM分数的候选把大量楼梯和天花板标成恐龙。增加背景负点、选取更紧致候选后，已生成的部分轮廓仍有明显误选，索引28甚至没有通过提示一致性的候选。**这些轮廓不具备直接投入可信Visual Hull消融的质量。** 这不是训练效果结论，而是输入标注质量问题。

已向用户补问，用户选择：**继续trex，先做恐龙掩膜的交互点选/修正，检查通过再跑18组训练。** 因此补齐了交互编辑器。轮廓未全部保存检查前不启动18组训练，也不把当前不合格的自动轮廓视为已确认输入。

## 已写好的入口与验证范围

- `tools/prepare_trex_silhouettes.py`：独立SAM依赖、提示点、三候选、训练/测试掩膜分离、预览与来源指纹；CPU试验揭示了上述失败，不能称为轮廓准备成功。
- `tools/foreground_hull.py`：只用训练轮廓构建前景体；与原背景点云、VGGT点集做受控并集。
- `tools/run_syngs_scene_ablation.py`：预检、掩膜准备、24组索引、复用6组、新训18组、独立输出、阶段缓存和完成标记。
- `tools/summarize_syngs_scene.py`：全图PSNR/SSIM/LPIPS、恐龙区域PNG PSNR、逐阶段/去模块差值、曲线与同视角图。
- `sh/full_scene/run_trex_syngs_scene_ablation.sh`：环境包装。

CPU与静态验证已通过：脚本编译/Bash语法、8种开关组合、所有CLI开关和10000步控制、基础/VGGT点集前缀完全一致、H补点开关独立、轮廓交集与图像外未知区域逻辑。**真实SAM轮廓、真实VH几何质量及新18组GPU训练尚未通过或尚未运行。** 不将合成单元检查写成真实场景重建验证。

结果将写入`3DGS/GaussianObject/output/trex_syngs_scene_ablation_v1/`，包括`ablation.csv`、`ablation.md`、`metrics_curves.png`、`stages_v4.png`/`stages_v6.png`/`stages_v9.png`和24组模型索引。恐龙区域PNG PSNR是对保存且裁剪/量化后的8位图按评测轮廓计算，与全图浮点渲染指标分栏标注，不混为同一种协议。

## 后续启动顺序

先完成轮廓修正与检查，然后只做CPU构建/投影核对，确认恐龙轮廓体没有明显落在背景，最后再启动18组训练。训练器本身、已有B和VD模型及其原指标均保持原状。

### 交互编辑器

新增`tools/review_trex_masks.py`和`tools/trex_mask_editor.html`。支持正/负提示点、限定框、三个SAM候选、画笔补选/擦除、撤销、逐张保存。浏览器服务仅监听远程127.0.0.1，通过SSH端口转发访问，没有训练接口。

在Mac本地终端执行（使用已有SSH免密配置）：

```bash
ssh -t -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 autodl-thesis 'cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA && source .venv-ddgs-vggt/bin/activate && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python 3DGS/GaussianObject/tools/review_trex_masks.py'
```

浏览器打开`http://127.0.0.1:8765`。先检查9张训练图，再检查7张评测图；绿色仅覆盖恐龙骨架，不覆盖楼梯、栏杆和天花板。默认自动轮廓不可靠，必要时清空提示点重新点选，而不是直接保存默认结果。最后点击“完成全部轮廓检查”，生成有来源与SHA的轮廓清单。人工修正耗时不确定，建议预留20–60分钟，复杂细骨骼可能更久。

逐张保存会核对轮廓尺寸/面积，清除本次尚未训练的旧几何缓存。轮廓与源图SHA不匹配则撤销检查状态；已有训练日志后拒绝修改同一版本输入。旧B、VD以及原始RGB不被覆盖。自动准备脚本会保留标记为人工检查的轮廓，避免重跑时覆盖画笔修改。

完成后在Mac终端Ctrl+C关闭编辑器（释放SAM显存），重新连接远程，在激活环境后先做CPU几何检查：

```bash
bash /root/autodl-tmp/Sparse-3DGS-heritage-VQA/3DGS/GaussianObject/sh/full_scene/run_trex_syngs_scene_ablation.sh --stage hulls --accept-masks
```

投影图为`data/LLFF/trex_syngs_scene_ablation_v1/hulls/v{4,6,9}/hull_projection_first_train.jpg`。轮廓体异常或点数不足时程序报错，不能把fallback当作H有效运行。人工核对投影后再在远程终端启动：

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
mkdir -p 3DGS/GaussianObject/output
nohup bash 3DGS/GaussianObject/sh/full_scene/run_trex_syngs_scene_ablation.sh --stage run --accept-masks > 3DGS/GaussianObject/output/trex_syngs_scene_ablation_v1.console.log 2>&1 < /dev/null &
tail -f 3DGS/GaussianObject/output/trex_syngs_scene_ablation_v1.console.log
```

此命令复用6个已完成结果，不重训；18个新增组各保留最终模型、7张测试渲染、指标和日志，不生成视频/优化器快照。新实验有独立输出，不删除旧模型。只在24组完整结果齐备后写`COMPLETE.json`。
