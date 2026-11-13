# CamixDirectorTraining — 有记忆的多机位音视频导播训练框架

## 源码仓库与本地数据

仓库只发布源码、启动脚本、配置模板、测试和使用文档。**不包含原始视频/音频、训练数据包、模型权重、虚拟环境、本地会话和实验报告**。克隆后请按安装说明创建环境，再自行准备素材；文中 `data/`、`runs/`、`reports/` 等路径是运行后生成的本地目录，历史实验数字不是随仓库附带的可直接复现结果。

下载、训练和导出需在本机执行，不会自动租用云GPU或接管Camix。软件可运行与模型达到专业导播质量是两件事，所有研究结果均未生产批准。

## 当前版本：Studio 0.3（本轮仅收敛Mac第一、第二优先级）

新增同步原片审片、点击偏好/多可接受/不确定标注、内容绑定审核、数据质量与不可变版本、微调/完整状态恢复、mmap按需读取、调度/早停/梯度累积、资源监控、实验对比/血缘、跨事件评估、离线闭环与候选门槛。

**从这里开始：[完整工作流说明](STUDIO_GUIDE.md)。** 本轮不改Camix App，不处理Windows，也不把界面升级当作模型质量升级。所有模型仍未生产批准。

下方0.2说明及历史实验报告保留作追溯；涉及“没有标注器/没有续训/全量载入内存”的旧限制以本节和Studio指南为准。

---

## 新增：本地网页训练台（Mac / Windows）

现在可用浏览器配置、启动／停止训练、查看日志／曲线、预览数据、运行评估和下载权重。

- **Mac**：运行 `bash start_web.command`。
- **Windows**：解压迁移包后，先运行 `setup_windows.bat`，再运行 `start_web_windows.bat`；CPU默认，CUDA按安装说明另选。
- 使用说明：`WEB_CONSOLE.md`；Windows环境与限制：`WINDOWS.md`。
- Windows不安装CoreMLTools；训练后将`.pt`交给Mac导出Core ML。Windows真机／CUDA尚未实测，不能把本机测试当作Windows验收。
- 当前没有人工偏好标注编辑器、优化器断点续训或大规模流式数据加载器；这些不作为本次已交付功能宣传。

---

**本工程独立于 Camix：不会编译、修改或控制 Camix，不会向 iPad 下发切播命令。**

当前交付的是可以继续训练、评估、导出的小型实验基线，**不是已经具备“各种场景自动导播能力”的大模型**。训练结果和限制见 `reports/RESULTS.md`。所有权重均标记 `production_approved=false`；输出只供离线分析或未来影子模式观察。

## 1. 这次实现了什么

- PyTorch 原始 RGB + PCM 双编码器，同一时刻的多机位注意力、跨时间的 Transformer，以及显式输入／输出的有限记忆。
- 不设置“会议／体育／婚礼”等场景分类，不预测固定机位编号类别。
- 共享机位评分头输出 `keep_value`、各机位 `take_value` 和 `switch_gain=take_value-keep_value`，通过排序偏好训练。
- 同步媒体导入、无标签掩码、按场次划分数据、训练检查点、独立测试、记忆消融、Core ML 导出和多步数值对齐检查。
- 公开 AMI 素材的小样本实验、可复现的合成烟雾测试，以及接入自己素材的模板。

**不分类不等于不需要监督。** 模型仍需要导演示范／人工偏好／可信反馈。AMI 的说话人词时间只提供弱监督，不会自动教会构图、美感、叙事、节奏或跨场景常识。

## 2. 快速使用

本机环境是 Apple Silicon Mac、Python 3.11、PyTorch 2.7.0、Core ML Tools 9.0、PyAV 16.0.1。依赖隔离在本目录 `.venv`，不修改系统 Python。首次在其他电脑使用时需重新创建虚拟环境；不要直接复制本机 `.venv` 到其他系统。

在终端中：

```bash
cd "$HOME/Desktop/CamixDirectorTraining"
bash setup.command
bash test.command
# 下载器会要求确认 AMI 信号和转写的 CC BY 4.0 条款；已有数据会校验后复用。
bash download_ami.command
bash train_ami.command
bash export_coreml.command
```

- `runs/LATEST_AMI_RUN.txt` 记录最近成功的 AMI 训练目录。
- 训练：`best.pt`、`training-report.json`。
- 评估：`evaluation/evaluation.json`、逐时刻 `value-trace.jsonl`，并不执行切换。
- 导出：`coreml_时间戳/ExperimentalDirectorStep.mlpackage` 和 `export-report.json`。
- 再次训练使用新目录，不覆盖已有权重；中断的实验保留日志后另开目录。
- 默认 CPU，PyTorch CLI 可选 `--device mps` 或 CUDA（相应环境需另建兼容依赖）。此交付只验证本机 CPU；不会自动使用或收费租用云 GPU。

## 3. 数据来源与训练范围

AMI Meeting Corpus 提供公开的同步多机位会议信号、近讲麦克风音轨及人工转写。出处、授权和改动说明见 `LICENSES_AND_DATA.md`；精确 URL、机位／麦克风映射、截取区间、转写归档校验值见 `data/ami_pilot/source-report.json`。

本次为三场会议各截取约 45 秒，分别作为训练／验证／测试；每场只选两名参与者的同步相机和音轨。保留原时钟的短音频窗口与低分辨率画面，避免占用数 GB 原片空间。官方映射不能简单用 Closeup 数字推导 Headset 数字。

**明确的局限：**

1. 目标来自人工转写的说话人时间，不是专业导演切镜头标签。
2. 当前节目机位是模拟的固定初始机位，不是真实导演的 PGM 历史。
3. 近讲麦克风不同于 Camix 两台手机在房间内收音，分布差异很大。
4. 数据很少、场景窄；即使弱标签准确率高，也不能等同于自动导播效果。
5. 不会把没标注的时刻自动当成 KEEP；重叠／其他人发言等不确定时刻不应强行提供监督。
6. 合成数据只验证张量／梯度／训练／导出链路，不能替代真实导播数据。

## 4. 使用自己的素材继续训练

推荐先收集 **2～4 机位、同一事件、同一时间轴** 的短片段。压低训练副本的分辨率／帧率，保留原始时间对齐和音轨，不必把高码率原片都放进训练目录。

1. 把 `examples/session.template.json` 复制到素材目录，按实际填写机位路径、时钟对齐、来源、场次 ID 和 split。
2. 每个训练小片段上限 120 秒；同一原始事件的多个片段必须使用同一个 `session_id` 并放在同一 split。**不要把同一录制的不同几秒随机分到训练和测试。**
3. 校验视频／音频对齐后才设 `alignment_verified=true`；标签确经人工审核后才设 `human_reviewed=true`。模板默认 false，不冒充已审核。
4. `program-events.jsonl` 记录真实已提交的节目切换（不是预览点击）；`preferences.jsonl` 记录导演在特定时刻更偏好哪个候选／保持。示例内容必须替换，不要原样当真值。
5. 用对应配置转换并训练：

```bash
.venv/bin/python -m camix_director.prepare /绝对路径/素材/session.json \
  --config examples/pilot-config.json --output data/custom/prepared/session_a.npz
# 另准备不同原始场次的 validation/test pack，然后：
bash train_custom.command /绝对路径/prepared /绝对路径/config.json
```

标注格式支持成对偏好：

```json
{"time_s":5.2,"preferred":"cam_b","rejected":["keep","cam_c"]}
{"time_s":8.0,"preferred":"keep","rejected":["cam_b"]}
```

也支持相对连续分值：`{"time_s":10.0,"scores":{"keep":0.2,"cam_b":0.8}}`。分数在当前损失中只用于**大小关系**，不拟合绝对质量值；未标注候选为 NaN。当前正在播的机位与 KEEP 是同一个动作，不重复训练。

`video_delay` / `audio_delay` 是在已对齐源文件上的额外可用性延迟（秒，0～1）；不是自动校准时钟漂移的工具。所有源的时间偏移／采样漂移必须先校准，未来实时适配也要维护一致时钟。

评估自己的权重：

```bash
.venv/bin/python -m camix_director.evaluate --checkpoint runs/你的实验/best.pt \
  --data data/custom/prepared --output runs/你的实验/evaluation
.venv/bin/python -m camix_director.export --checkpoint runs/你的实验/best.pt \
  --output runs/你的实验/coreml_fp32 --precision fp32 --parity-steps 12
```

更大配置 `examples/standard-config.json` 是**未训练、未验收的扩容起点**：4 个机位、128 像素、1 秒 PCM、128 维／4 层 Transformer、32 个记忆 token、可选 ImageNet 预训练 MobileNetV3。首次启用预训练视觉编码器会下载官方权重；不代表已经有多场景导播预训练权重。它与当前 pilot 权重形状不同，需重新训练，不能直接互换。

## 5. 实时音视频怎样输入、怎样记忆

模型不接收 RTMP/RTC URL，也不解复用网络流。未来流适配器先解码、同步、补齐固定机位槽，再每 200ms 调用一次模型：

```text
多路解码音视频 → 统一时钟 → 最近的过去帧 + 截止当前时刻的 PCM 窗口
              → 模态/机位有效掩码 → MemoryDirector.forward(..., memory, memory_mask)
              → KEEP/TAKE 数值 + 新记忆 → 下一步继续使用
```

本次 pilot 每路视频是 `[3,64,64]`、PCM 是 16kHz 单声道最近 4000 个采样（250ms），5Hz 决策；这不是要求直播画面也降为 64 像素或每秒 5 帧。完整输入／输出见 `docs/MODEL_CONTRACT.md`，Python 顺序推理边界见 `camix_director/stream.py`。`StreamingPolicy` 要求单调时间戳，房间／机位代次变化时清空记忆，拒绝倒退和重复帧时间。

记忆是最近 8 个学习得到的**全局摘要 token**，显式窗口约 1.6 秒，并带递归摘要信息；它不是长期完整录像、不是每机位独立长期记忆，也不是逐层 KV cache。不会无限增长，也不会看到未来帧。

实时延时包括采集／编码／网络／解码／对齐等待／模型／切播执行。报告中的 Mac CPU 单步耗时只测模型，不代表 iPad 总链路延迟。250ms PCM 窗口使用已经到达的过去采样；不意味着每次额外等待 250ms。模型持续在真实多路流上运行仍需 Camix 端适配和真机测试，本轮未接入。

## 6. 值怎么解释

- `keep_value`：保留当前节目机位的相对偏好值。
- `take_value[i]`：切到机位 i 的相对偏好值。
- `switch_gain[i]`：与保持相比的差值；正值只表示模型更偏好该候选。
- 无效机位 TAKE=-10000，仅作屏蔽；不是实际质量分。
- **这些值不是置信度／概率，也不是统一可迁移阈值。** 排序训练的共同偏置不唯一，不能把 0.8 解释成 80% 正确。
- 当前权重 `execute_allowed=false`，不会因为正差值直接发 TAKE。未来即使上线也需独立的离线校准／影子验证、人工接管、信号失效保护、最短镜头时长、切换成本和闭环测试。

## 7. 文件导航

- `camix_director/model.py`：原始音视频编码器、跨机位／时间注意力、显式记忆、数值头。
- `data.py` / `prepare.py`：数据契约、场次隔离、自己的素材导入。
- `ami.py` / `media.py` / `download.py`：公开数据、过去帧／PCM 采样、校验下载。
- `train.py` / `evaluate.py`：序列偏好训练、最佳验证权重、独立测试／记忆消融。
- `export.py`：Core ML 导出、多步独立记忆滚动数值验证。
- `tests/`：可离线运行的回归测试。
- `reports/`：本机结果与空间清理记录。
- `downloads/`：已核验的依赖 wheel 缓存；可删除后重新联网下载。`data/` 是准备好的训练副本，`runs/` 是权重与报告，**不要当缓存盲删**。

本工程未擅自宣布开源。第三方数据与依赖仍受各自许可约束；对外发布或商用前请审核来源、个人权利和归属要求。

数据预处理质检（画面联系表／音轨有效率／标注比例）：

```bash
.venv/bin/python -m camix_director.inspect_data --data data/ami_pilot/prepared \
  --config examples/pilot-config.json --out reports/data-qa --allow-weak-supervision
```

程序质检与源发布的同步说明不能替代人工检查自己素材的时钟；导入器不会自动发现每一种音画漂移。

## 8. 本次迭代和新的留出测试

第一版 `runs/ami_pilot_20261112` 使用未归一化 PCM 编码，诊断发现几乎总偏向切到另一机位（ES2004a：51.7%，与该片段的“总切换”基线相同），不能算有效学习。

第二版 `runs/ami_normalized_balanced_20261112` 保留原始音频，内部增加波形增益归一化与可学习的对数能量特征；用训练集统计对 KEEP-better / TAKE-better 排序对做重要性加权，防止“始终切换”的多数偏差。它仍是排序损失，不是场景分类或人工音量阈值规则。完整对照见结果报告。

V2 在查看 ES2004a 诊断后设计，因此 ES2004a 不再作为“从未见过结果的盲测”宣传。V2 根据 ES2003a 验证损失选定后冻结权重，再额外准备 ES2005a 作为新留出测试；冻结哈希和时间在 `reports/fresh-holdout-protocol.json`。该留出仍只是很短的会议说话人代理任务，不是产品验收集。以后反复查看它的结果或根据它调参后，也应更换新的独立场次。

```bash
# 已有新留出数据时，对选定的检查点评估（不训练）：
.venv/bin/python -m camix_director.evaluate \
  --checkpoint runs/ami_normalized_balanced_20261112/best.pt \
  --data data/ami_fresh_holdout/prepared \
  --output runs/ami_normalized_balanced_20261112/evaluation_fresh --allow-weak-supervision
```

评估包含：完整模型、每步清空记忆、静音音频、抹掉画面内容、简单两机位 PCM-RMS 基线，以及 KEEP/TAKE 两类偏好对的平衡准确率。去掉画面／音频后分数不降，就不能宣称模型已依赖该模态；当前会议数据也不能验证画面构图能力。验证集训练窗口有重叠，窗口内比较次数不等于独立样本数。
