# v2: 稀疏3-view / 可选9-view梯度诊断

默认horns、fortress，3-view，每方法10000步。3个相机来自原成功9-view相机列表的首/中/末（索引0、4、8）；此前没有独立3-view配置，本次不冒充已存在的3-view相机设定。v9沿用原完整9个相机。均从排除LLFF holdout(idx%8==0)的相机池选取，不用测试图像loss统计训练梯度。源码与输出隔离，seed0，统计step1000..4950每50步（标准10000步配置）。两方法相机/迭代序列逐项匹配才绘图。

## 运行
有GPU后：bash 3DGS/reconstruction/experiments/gradient_depth_drop/start.sh --views 3
可选另跑：同命令 --views 9
CPU检查：CUDA_VISIBLE_DEVICES='' bash 3DGS/reconstruction/experiments/gradient_depth_drop/start.sh --check --views 3
只重画：同命令 --plot-only --views 3，不加载GPU。

## 路径
output/gradient_depth_drop_v1/prepared/{horns,fortress}：共享既有去畸变图像，不复制媒体。
output/gradient_depth_drop_v1/v3/{horns,fortress}：主实验。
output/gradient_depth_drop_v1/v9/{horns,fortress}：可选补充，绝不混入v3。
原9-view结果移到output/gradient_depth_drop_previous9v_20261120，当前旧阈值计数无法恢复，不能伪造。旧prepare数据移到新共享prepared，避免重复。旧源run.py保存在归档source_run_v1.py。新统计不复用旧模型/均值，也不自动运行GPU。

## 梯度
保留实际loss.backward后、增密前viewspace_points.grad[:,:2].norm；有效=visibility_filter且正Z且有限。可见零梯度（包括Drop导致的零梯度）保留。单位Gaussian-视角-训练步观测，不是独立/去重Gaussian数量。
按共同初始点云、逐相机正Z min/max固定归一化，训练中范围不重新拟合。两方法depth_bounds.json必须一致；范围外clip并计数。全10个0.1箱都保留，不筛选有利距离。

## 显示
horns xlim0..0.7，fortress0..0.3。0.05只是刻度，不是新箱；每个0.1箱中点绘制。保留0..0.1；远区原始数据继续保留。CSV空箱均值/比例明确NaN，有样本但零梯度均值保留0。真实图PNG600dpi与PDF，蓝虚线圆点vs红实线方块。轴按用户给定英文，结果解释中文；far是可视范围右端，不表示重新定义归一化最远为0.7/0.3。

## 输出
gradient_bins.csv：各方法count,sum,mean,threshold,above_count,above_ratio,zero_count；瞬时严格>实际项目阈值（标准参数0.0002）。
densification_threshold_bins.csv：在真正触发增密检查时、add_densification_stats之后和densify_and_prune之前，按xyz_gradient_accum/denom>=threshold计数（项目clone/split用>=）。只统计denom>0且正Z；按当前检查相机的深度分箱。该量累积多个训练步和相机的梯度，不是当前一步瞬时梯度。计数为Gaussian-增密检查观测，会重复出现，不等于新生成Gaussian数量。clone/split还检查尺度，prune还检查opacity等条件；本统计不声称等于最终存活的新增点数。
每方法statistics.json包含逐统计时刻数量/和/超阈计数，以及全体正Z候选逐箱数量与可见后数量、范围外裁剪数。可区分初始范围被离群点拉大、可见性筛选或远区候选少。不是按新结果改分位数或截掉不利样本。
per_observation_bins.csv保留每一步每箱；bin_comparison.csv及interpretation.md自动解释平均与瞬时超阈比例的升/降和样本少的情况，不预设Drop获胜。密度累计阈值另表，不和瞬时指标混为一谈。
3DGS/drop子目录保留最终render/GT、逐测试视角PSNR/SSIM/LPIPS；测试仅最终no_grad评测，不参与梯度统计。

## 证据边界
旧9-view数据显示Drop在两个场景的部分近区平均梯度下降，不支持所有区间均改善。新3-view实际曲线/阈值数据须GPU跑后才能解释。更大梯度/更多超阈点不自动代表重建改善；需与渲染指标共同看，且单种子、重复观测不提供独立统计显著性。不能为了曲线更好而择范围或重新归一化。未重新做3-view独立COLMAP，本实验沿用数据集全视角预标定与初始化，需在论文说明。

CPU测试仅分箱接口合成数组，使用临时/单元检查，不生成假实验结果。实际GPU链路待用户启动。
