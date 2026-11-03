# 连续字幕2048：val90固定配置补充验证
预算、相邻句规则、时间标记、提示词、32帧和像素预算全部沿用冻结配置。
val90此前已参与诊断，本轮是补充验证，不能称为新的独立测试。test225不动；不涉及Agent、API或训练。

## 执行范围
90题、30个视频；三路270个逻辑答案。原无字幕与原对齐字幕的180个旧答案，仅在重建的实际输入token哈希、原文和视觉布局与旧记录一致且原生输入检查通过后复用；不符则停止，禁止自动补跑基线。最多90次新生成，仅用于连续字幕；缺字幕时回退同题无字幕回答，不减少分母。
旧日志没有保存历史像素tensor，不能声称直接比较过历史像素缓存。当前输入tensor与当前原生处理逐字段核验；旧来源通过视频、协议、帧/秒数、输入token哈希及视觉记录锁定。

## 保存与断点
入口：VQA/mage_vl/practical/run_continuous_val90.sh。
所有成果在VQA/eval_results/practical_mage_v1/：
15_continuous_val90/保存候选、CPU检查、logs完整日志和val90_t2048/summary.json、predictions.json；
cache/continuous_val90/<签名>/保存协议、环境、证据及逐答案JSONL断点；
07_archive/continuous_val90_fixed2048_v1.json在汇总完成后封存。
中断后原命令重跑，跳过已封存单元；来源冲突或损坏停止并保留现场。重复运行已完成阶段不加载模型。
不保存帧、视频、tensor或新模型权重，不删旧缓存。入口及运行过程中检查空间，默认至少保留10GiB，整个practical_mage_v1目录限制1GiB。

## 执行命令
先激活Mage环境、进入工程并确认vlm，然后设置PRACTICAL_STAGE_CONFIRMED=1。
依次执行bash VQA/mage_vl/practical/run_continuous_val90.sh init、run、summarize。
CPU初始化已完成，没有启动推理。tqdm显示270个逻辑答案，其中180个是旧答案输入核查与复用。
结果重点看中等/长视频的救回、误伤、净增和视频聚类置信区间。即使结果为负也保留，不自动改预算、不按时长选答案，不自动扩跑test或全量。
