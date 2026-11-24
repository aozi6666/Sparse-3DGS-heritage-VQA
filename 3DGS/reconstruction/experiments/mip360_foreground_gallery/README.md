# Mip360 三物体前景训练阶段可视化

独立入口：`experiments/mip360_foreground_gallery/start.sh`。输出：`output/mip360_foreground_gallery_v1/`。
源数据、掩膜和既有模型保持只读；不修改已有训练入口。

```bash
cd /root/autodl-tmp/Sparse-3DGS-heritage-VQA
source .venv-ddgs-vggt/bin/activate
bash 3DGS/reconstruction/experiments/mip360_foreground_gallery/start.sh --check
nohup bash 3DGS/reconstruction/experiments/mip360_foreground_gallery/start.sh > 3DGS/reconstruction/output/mip360_foreground_gallery_v1.console.log 2>&1 < /dev/null &
tail -f 3DGS/reconstruction/output/mip360_foreground_gallery_v1.console.log
```

启动脚本自动激活环境。场景顺序bonsai、garden、kitchen，单独子进程串行运行。
总日志显示场景阶段；详细进度分别在输出根目录的bonsai.log、garden.log、kitchen.log。
完成场景有complete.json，重复命令会跳过；部分场景保留现场，重跑需新的`--out`，不是优化器断点续训。
同一输出目录有进程锁，输入划分、相机、掩膜和脚本保存哈希，配置变化拒绝沿用。

每个物体读取现有sparse_9与sparse_test划分。只使用九张训练图的掩膜筛选既有SfM点：至少两个前景支持，且前景支持占可见训练相机比例至少80%。
保留的是物体初始化，不走repair、LoRA、VGGT或Drop。现有全序列SfM点及相机是预先已知的，**此流程是可视化示例，不能当成严格只用九张原图的独立重建消融。**
输入RGB、mask通过符号链接共享；相机BIN与划分复制进隔离目录。不会在原始sparse目录生成PLY。
白背景掩膜训练10000步，固定种子0、SH2、原标定尺寸的1/4；20万高斯是停止加密的软阈值，不是绝对上限。
展示视角在训练前按测试掩膜前景占比最大选定，不能根据方法增益事后挑选。

候选图在第10、25、50及每100步采集，只缓存在内存中。
优先在最佳PSNR附近15.3 dB内，按实测PSNR/SSIM/LPIPS的联合距离分散选九张（不足时扩大候选窗口）。不修改画面凑指标，不保证达到截图跨度，不给阶段图套用其他方法名称。
候选、最终导出图都直接渲染于白背景，**预测结果没有使用测试GT掩膜重新合成背景**，背景漏点会进入误差。
输出GT用同一测试视角掩膜和白背景；GT与渲染均量化至8位再评价。

每场景输出：
- 01_iterXXXXX.png ... 09_iterXXXXX.png：按实测PSNR降序排列的9个不同阶段。
- GT.png：同一测试相机的前景GT；三个场景共30张独立PNG。
- overview.jpg：9张图＋GT的拼图，标注真实步数及全图指标。
- metrics.csv：全图PSNR、SSIM、VGG LPIPS、LPIPS×100；前景像素PSNR、固定主体裁剪区域SSIM/LPIPS。
- evaluation_region.json：固定裁剪框、前景比例与评价定义。裁剪框为GT掩膜外接框向外扩12像素。
- candidates.json：全部采集阶段的实测指标。
- complete.json：实际跨度及与截图参考范围是否相符。
- protocol.json：训练/测试照片、展示相机、初始化点数及协议。
- model/：仅保留最后一个模型；无视频、优化器检查点、TensorBoard图片或大量中间模型。

参考截图整列跨度：PSNR 11.72–15.29 dB、SSIM 0.092–0.117、LPIPS原值0.1574–0.1719。这些是方法表跨度，不能要求单个场景、单个视角精确复现。若未满足，complete.json明确记为false。
全图指标可能受白背景占比影响，尤其garden主体较小。前景PSNR与裁剪SSIM/LPIPS必须与全图指标分别标注，不能混为同一评测协议。
阶段图、单视角指标不能冒充多方法对比或多场景平均结果。

CPU预检及真实前景初始化测试已通过：
- bonsai：292张，9训练/37测试，23515个初始化点；展示相机DSCF5677。
- garden：185张，9训练/24测试，2302个初始化点；展示相机DSC08012。
- kitchen：279张，9训练/35测试，36519个初始化点；展示相机DSCF0888。
GPU训练与渲染待用户启动。运行前至少4 GiB空闲，采图时低于2 GiB中止。
