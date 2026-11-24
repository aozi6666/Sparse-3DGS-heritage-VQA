# Omni深度独立测试入口

只在用户明确运行run时加载GPU；check仅CPU只读，prepare只CPU生成独立索引。不会动正在复制的原数据、既有训练清单/权重/损失，不下载。

原始数据VGGT-Object/data/omni3d_depth。默认100物体、每物体24视角，从102候选按类别轮转、类别内seed42固定选取。需要整个候选目录的声明视角都非零、格式和相机校验通过，不能仅按文件名计数。prepare进一步实际读取选中深度，RGBA前景alpha>127、深度有限且0<d<65504，RGB与深度尺寸一致，各视角至少100有效像素。文件r_1只匹配r_1_depth.exr/r_1.exr及NPY同名候选，不误配r_10。

65504原生背景先过滤再ray→Z，防止除以射线长度后背景变成有限前景。--depth-type必须显式选择camera_z_depth或ray_distance，未核实不猜；场景单位不冒称米。已有Omni转换入口prepare.py也复用精确匹配与背景过滤，其余数据/损失不变。

## 现在可运行的只读检查

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B VGGT-Object/tools/omni_depth_eval.py check
```

上传未完成返回2，打印类别和缺失示例，不产生索引，也不自动等待后启动GPU。

## 文件完整后再冻结计划

```bash
python -B VGGT-Object/tools/omni_depth_eval.py prepare --depth-type VERIFIED_NATIVE_TYPE
```

VERIFIED_NATIVE_TYPE须替换为已验证的camera_z_depth或ray_distance，不能把占位符原样执行。计划默认VGGT-Object/results/omni_depth_test_v1/plan.json，存在则拒绝覆盖。索引直接引用原文件，冻结选中RGB/EXR/相机SHA256及评测代码SHA256，拒绝与现有训练/验证清单ID重叠。原始102物体不会移动，也不重复生成RGB/NPY。

## 用户明确安排GPU后

```bash
python -B VGGT-Object/tools/omni_depth_eval.py run
```

无--adapter时仅原始VGGT，所有参数冻结，不训练。默认为392×392，每个物体24视角分6组，每组4视角，不是24视角联合推理。输入原始权重3DGS/reconstruction/third_party/vggt/models/model.pt。未来微调对照可显式--adapter对应best，并用同一--plan、全新--out；不会自动使用A16或旧400步权重。run重算输入/代码SHA，复制后改变或源码漂移拒绝复用；输出存在拒绝覆盖，无自动断点续跑，也不排队。

AbsRel/RMSE/RMSE_Log/delta同时保留GT定义规范尺度与每组跨4视角共享median对齐两种口径；前者不是米制绝对误差。固定GT mask，不因模型错误过滤像素；非有限/非正预测直接停止。先每对象平均6组，再对象宏平均。

Boundary F1沿用log深度跳变>0.05、2像素容差的前景内部边界定义，不包含外轮廓。GT无边界时null并报告覆盖数；GT有边界预测无边界时0，不能静默记null而隐藏漏检。该规则仅在新独立测试入口中应用，旧历史指标实现和历史结果不修改。没有外轮廓指标/跨数据集官方可比性宣称。

结果目录默认VGGT-Object/results/omni_depth_test_v1/original，含协议、逐对象逐组JSON、per_object_metrics.csv、summary.json、complete.json。只保存指标不复制模型/原始深度，不做3DGS训练或渲染实验。

本次CPU合成测试只验证接口/分箱外的数值规则，不算真实原版VGGT成绩。GPU推理、100对象像素验收与本地文件来源/几何语义确认仍待用户完成上传后进行。
